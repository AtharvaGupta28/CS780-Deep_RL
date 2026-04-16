
import sys, os, random, time, json
from collections import deque
import numpy as np

REPO_PATH = '/content/drive/MyDrive/CS780-OBELIX'
SAVE_DIR  = '/content/drive/MyDrive/CS780-OBELIX'

sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
os.system("pip install -r requirements.txt -q")
from obelix import OBELIX

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
L45, L22, FW, R22, R45 = 0, 1, 2, 3, 4
ROTATIONS = {0, 1, 3, 4}

P_INIT      = 'INIT'
P_SEEK_WALL = 'SEEK_WALL'
P_ALIGN     = 'ALIGN'
P_SEARCH    = 'SEARCH'
P_FIND_GAP  = 'FIND_GAP'
P_CROSS_GAP = 'CROSS_GAP'
P_APPROACH  = 'APPROACH'
P_PUSH      = 'PUSH'

SEARCH_ROW_DEPTH     = 42
SEARCH_LATERAL       = 14
SEARCH_N_ROWS        = 9
BOUNDARY_FLIP_THRESH = 3
GAP_FREE_THRESHOLD   = 10
GAP_CROSS_STEPS      = 32
INIT_SCAN_TURNS      = 8

N_BITS   = 8
N_STATES = 2 ** N_BITS
HAMMING_WEIGHTS   = np.array([1,1,1,1,1,1,4,4], dtype=np.float32)
HAMMING_THRESHOLD = 1

TOTAL_EPISODES   = 5000
MAX_STEPS_NOWALL = 700
MAX_STEPS_WALL   = 450
ALPHA            = 0.22
ALPHA_NEIGHBOUR  = 0.06
GAMMA            = 0.96
LAMBDA           = 0.85
TEMP_START       = 4.0
TEMP_END         = 0.4
TEMP_DECAY       = 0.9975

DIFFICULTY_SCHEDULE = [(0, 0), (1500, 2), (3000, 3)]
WALL_SCHEDULE = [(0,0.0),(1000,0.15),(2000,0.35),(3500,0.55),(4500,0.65)]

SAVE_BEST_SR    = os.path.join(SAVE_DIR, 'q_map_best_sr.npy')
SAVE_BEST_SR_NW = os.path.join(SAVE_DIR, 'q_map_best_sr_nowall.npy')
SAVE_LATEST     = os.path.join(SAVE_DIR, 'q_map_latest.npy')
SAVE_LOG        = os.path.join(SAVE_DIR, 'training_log_map.json')

print("Precomputing Hamming neighbours (256 states)…")
_t = time.time()
_ALL_BITS = np.array(
    [[(s >> b) & 1 for b in range(N_BITS)] for s in range(N_STATES)],
    dtype=np.float32)
_NEIGHBOUR_MASK = np.zeros((N_STATES, N_STATES), dtype=bool)
for s in range(N_STATES):
    d = np.abs(_ALL_BITS - _ALL_BITS[s]) @ HAMMING_WEIGHTS
    _NEIGHBOUR_MASK[s] = (d <= HAMMING_THRESHOLD) & (np.arange(N_STATES) != s)
print(f"  Done in {time.time()-_t:.2f}s | "
      f"avg neighbours: {_NEIGHBOUR_MASK.sum(1).mean():.1f}")

ACTION_ORIENT_DELTA = {L45: 2, L22: 1, FW: 0, R22: -1, R45: -2}

def update_orient(orient, action_idx):
    return (orient + ACTION_ORIENT_DELTA[action_idx]) % 16

def turns_to_face(current, target):
    diff = (target - current) % 16
    if diff == 0: return []
    if diff <= 8:
        acts = [L45] * (diff // 2)
        if diff % 2: acts.append(L22)
        return acts
    else:
        diff = 16 - diff
        acts = [R45] * (diff // 2)
        if diff % 2: acts.append(R22)
        return acts

def build_search_plan(wall_orient, current_orient, along_dir=None):
    
    away_dir  = (wall_orient + 8) % 16
    wall_dir  = wall_orient
    if along_dir is None:
        along_dir = (wall_orient + 4) % 16

    seq   = []
    seq  += turns_to_face(current_orient, along_dir)
    track = along_dir

    for row_i in range(SEARCH_N_ROWS):
        if row_i > 0:
            seq += [FW] * SEARCH_LATERAL

        seq += turns_to_face(track, away_dir);  track = away_dir
        seq += [FW] * SEARCH_ROW_DEPTH
        seq += turns_to_face(track, wall_dir);  track = wall_dir
        seq += [FW] * SEARCH_ROW_DEPTH
        seq += turns_to_face(track, along_dir); track = along_dir

    return seq


def build_gap_cross_plan(wall_orient, current_orient):
    away_dir = (wall_orient + 8) % 16
    seq  = turns_to_face(current_orient, away_dir)
    seq += [FW] * GAP_CROSS_STEPS
    return seq

def make_mem():
    return {
        'phase':          P_INIT,
        'phase_step':     0,
        'orient':         0,
        'wall_orient':    None,
        'along_dir':      None,
        'action_seq':     [],
        'seq_idx':        0,
        'search_boundary_hits': 0,
        'prev_stuck':     False,
        'free_streak':    0,
        'gap_found':      False,
        'gap_crossed':    False,
        'consec_stuck':   0,
        'consec_stuck_push': 0,
        'pushing':              False,
        'recently_visible':     False,
        'steps_since_visible':  0,
        'last_seen_left':       False,
        'last_seen_right':      False,
        'drift_dir':            None,
    }


def update_mem_approach(mem, obs, raw_reward):
    if raw_reward >= 90:
        mem['pushing'] = True
    any_s = np.any(obs[:17] > 0.5)
    lnow  = np.any(obs[4:6]  > 0.5) or np.any(obs[12:14] > 0.5)
    rnow  = np.any(obs[6:8]  > 0.5) or np.any(obs[14:16] > 0.5)
    if any_s:
        mem['steps_since_visible'] = 0
        mem['recently_visible']    = True
        if mem['last_seen_left']  and rnow and not lnow: mem['drift_dir'] = 'right'
        elif mem['last_seen_right'] and lnow and not rnow: mem['drift_dir'] = 'left'
        mem['last_seen_left']  = lnow
        mem['last_seen_right'] = rnow
    else:
        mem['steps_since_visible'] += 1
        if mem['steps_since_visible'] > 25:
            mem['recently_visible'] = False
            mem['drift_dir']        = None

def encode_approach(obs, mem):
    fn = int(np.any(obs[0:4]   > 0.5))
    ln = int(np.any(obs[4:6]   > 0.5))
    rn = int(np.any(obs[6:8]   > 0.5))
    ff = int(np.any(obs[8:12]  > 0.5))
    lf = int(np.any(obs[12:14] > 0.5))
    rf = int(np.any(obs[14:16] > 0.5))
    ir = int(obs[16] > 0.5)
    ps = int(mem['pushing'])
    return (fn|(ln<<1)|(rn<<2)|(ff<<3)|(lf<<4)|(rf<<5)|
            (ir<<6)|(ps<<7)) % N_STATES


def init_approach_qtable():
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for s in range(N_STATES):
        fn=(s>>0)&1; ln=(s>>1)&1; rn=(s>>2)&1
        ff=(s>>3)&1; lf=(s>>4)&1; rf=(s>>5)&1
        ir=(s>>6)&1; ps=(s>>7)&1
        if ir or ps:
            q[s,FW]=5.0; q[s,L22]=-0.5; q[s,R22]=-0.5
            q[s,L45]=-1.5; q[s,R45]=-1.5
        elif fn: q[s,FW]=3.0
        elif ff: q[s,FW]=2.0
        elif ln and not rn: q[s,L22]=2.0; q[s,FW]=0.8
        elif rn and not ln: q[s,R22]=2.0; q[s,FW]=0.8
        elif lf and not rf: q[s,L22]=1.5; q[s,FW]=0.5
        elif rf and not lf: q[s,R22]=1.5; q[s,FW]=0.5
        else: q[s,FW]=1.0
    return q


def shape_approach_reward(raw_r, obs, mem, action_idx):
    r     = float(raw_r)
    any_s = np.any(obs[:17] > 0.5)
    ir    = obs[16] > 0.5
    stuck = obs[17] > 0.5
    if any_s or mem['pushing']:
        if action_idx == FW:             r += 1.5
        if ir and not mem['pushing']:    r += 3.0
        if mem['pushing'] and not stuck: r += 0.5
    return r


def select_approach_action(state, q_table, temperature, obs, mem,
                            action_history, burst):
    if burst > 0:
        return FW
    if len(action_history) >= 8:
        if sum(1 for a in list(action_history)[-8:] if a in ROTATIONS) >= 7:
            if random.random() < 0.9:
                return FW
    q_vals = q_table[state].copy()
    if mem['recently_visible'] and not np.any(np.asarray(obs[:17]) > 0.5):
        if mem['drift_dir'] == 'right': q_vals[R22]+=0.5; q_vals[R45]+=0.3
        elif mem['drift_dir'] == 'left': q_vals[L22]+=0.5; q_vals[L45]+=0.3
    q_shifted = q_vals - np.max(q_vals)
    probs = np.exp(q_shifted / max(temperature, 0.05))
    probs /= probs.sum()
    return int(np.random.choice(N_ACTIONS, p=probs))

TRACE_THRESHOLD = 0.005

def update_with_traces(q_table, traces, state, action_idx, td_error):
    decay = GAMMA * LAMBDA
    dead  = [k for k, v in traces.items() if abs(v*decay) < TRACE_THRESHOLD]
    for k in dead: del traces[k]
    for k in traces: traces[k] *= decay
    key = (state, action_idx)
    traces[key] = max(traces.get(key, 0.0) + 1.0, 1.0)
    for (s, a), tv in traces.items():
        q_table[s, a] += ALPHA * td_error * tv
        q_table[_NEIGHBOUR_MASK[s], a] += ALPHA_NEIGHBOUR * td_error * tv

def decay_traces(traces):
    decay = GAMMA * LAMBDA
    dead  = [k for k, v in traces.items() if abs(v*decay) < TRACE_THRESHOLD]
    for k in dead: del traces[k]
    for k in traces: traces[k] *= decay

def fsm_step(mem, obs, q_table, temperature, action_history, burst, step):
    any_s = np.any(obs[:17] > 0.5)
    stuck = obs[17] > 0.5
    mem['consec_stuck'] = mem['consec_stuck'] + 1 if stuck else 0
    phase = mem['phase']

    if any_s and phase not in (P_APPROACH, P_PUSH):
        mem['phase'] = P_APPROACH; mem['phase_step'] = 0
        mem['action_seq'] = []; phase = P_APPROACH
    if mem['pushing'] and phase != P_PUSH:
        mem['phase'] = P_PUSH; mem['phase_step'] = 0; phase = P_PUSH

    mem['phase_step'] += 1

    if phase == P_INIT:
        action = R45 if mem['phase_step'] <= INIT_SCAN_TURNS else FW
        if mem['phase_step'] > INIT_SCAN_TURNS:
            mem['phase'] = P_SEEK_WALL; mem['phase_step'] = 0
        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    elif phase == P_SEEK_WALL:
        if stuck:
            mem['wall_orient'] = mem['orient']
            mem['along_dir']   = (mem['wall_orient'] + 4) % 16
            mem['action_seq']  = turns_to_face(mem['orient'], mem['along_dir'])
            mem['seq_idx']     = 0
            mem['phase'] = P_ALIGN; mem['phase_step'] = 0
            action = L45
        else:
            action = FW
        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    elif phase == P_ALIGN:
        if mem['seq_idx'] < len(mem['action_seq']):
            action = mem['action_seq'][mem['seq_idx']]
            mem['seq_idx'] += 1
        else:
            plan = build_search_plan(
                mem['wall_orient'], mem['orient'], mem['along_dir'])
            mem['action_seq']           = plan
            mem['seq_idx']              = 0
            mem['search_boundary_hits'] = 0
            mem['phase'] = P_SEARCH; mem['phase_step'] = 0
            action = FW
        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    elif phase == P_SEARCH:
        if stuck and not mem['pushing']:
            mem['search_boundary_hits'] += 1
            if mem['search_boundary_hits'] >= BOUNDARY_FLIP_THRESH:
                # Flip search direction 180°
                mem['along_dir']   = (mem['along_dir'] + 8) % 16
                mem['wall_orient'] = (mem['wall_orient'] + 8) % 16
                plan = build_search_plan(
                    mem['wall_orient'], mem['orient'], mem['along_dir'])
                mem['action_seq']           = plan
                mem['seq_idx']              = 0
                mem['search_boundary_hits'] = 0
            else:
                # Skip past current FW block
                skip = mem['seq_idx']
                fwd_seen = 0
                while skip < len(mem['action_seq']):
                    if mem['action_seq'][skip] == FW:
                        fwd_seen += 1
                        if fwd_seen > SEARCH_LATERAL: break
                    skip += 1
                while (skip < len(mem['action_seq']) and
                       mem['action_seq'][skip] != FW):
                    skip += 1
                mem['seq_idx'] = min(skip, len(mem['action_seq']))
            action = R45
            mem['orient'] = update_orient(mem['orient'], action)
            return action, False

        if mem['seq_idx'] < len(mem['action_seq']):
            action = mem['action_seq'][mem['seq_idx']]
            mem['seq_idx'] += 1
            if action == FW:
                mem['search_boundary_hits'] = 0
        else:
            along_dir = mem['along_dir']
            mem['action_seq'] = turns_to_face(mem['orient'], along_dir)
            mem['seq_idx']    = 0
            mem['prev_stuck'] = False; mem['free_streak'] = 0
            mem['phase'] = P_FIND_GAP; mem['phase_step'] = 0
            action = mem['action_seq'][0] if mem['action_seq'] else FW
            if mem['action_seq']: mem['seq_idx'] = 1

        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    # ── FIND_GAP ──────────────────────────────────────────────────────────
    elif phase == P_FIND_GAP:
        if mem['seq_idx'] < len(mem['action_seq']):
            action = mem['action_seq'][mem['seq_idx']]
            mem['seq_idx'] += 1
            mem['orient'] = update_orient(mem['orient'], action)
            return action, False

        if stuck:
            action = R45
            mem['prev_stuck'] = True; mem['free_streak'] = 0
        else:
            if mem['prev_stuck']:
                mem['free_streak'] += 1
                if mem['free_streak'] >= GAP_FREE_THRESHOLD:
                    cross = build_gap_cross_plan(mem['wall_orient'], mem['orient'])
                    mem['action_seq'] = cross; mem['seq_idx'] = 0
                    mem['gap_found']  = True
                    mem['phase'] = P_CROSS_GAP; mem['phase_step'] = 0
                    action = mem['action_seq'][0] if mem['action_seq'] else FW
                    if mem['action_seq']: mem['seq_idx'] = 1
                    mem['orient'] = update_orient(mem['orient'], action)
                    return action, False
            else:
                mem['free_streak'] = 0
            action = FW; mem['prev_stuck'] = False

        if mem['phase_step'] > 250 and not mem['gap_found']:
            mem['wall_orient'] = (mem['wall_orient'] + 8) % 16
            mem['along_dir']   = (mem['along_dir'] + 8) % 16
            mem['action_seq']  = turns_to_face(mem['orient'], mem['along_dir'])
            mem['seq_idx'] = 0; mem['phase_step'] = 0
            mem['free_streak'] = 0; mem['prev_stuck'] = False

        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    # ── CROSS_GAP ─────────────────────────────────────────────────────────
    elif phase == P_CROSS_GAP:
        if mem['seq_idx'] < len(mem['action_seq']):
            action = mem['action_seq'][mem['seq_idx']]
            mem['seq_idx'] += 1
        else:
            mem['wall_orient'] = None; mem['gap_crossed'] = True
            mem['prev_stuck'] = False; mem['free_streak'] = 0
            mem['search_boundary_hits'] = 0
            mem['action_seq'] = []; mem['phase'] = P_SEEK_WALL
            mem['phase_step'] = 0; action = FW
        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    # ── APPROACH ──────────────────────────────────────────────────────────
    elif phase == P_APPROACH:
        if stuck and not mem['pushing']:
            action = R45
            mem['orient'] = update_orient(mem['orient'], action)
            return action, False
        state  = encode_approach(obs, mem)
        action = select_approach_action(
            state, q_table, temperature, obs, mem, action_history, burst)
        mem['orient'] = update_orient(mem['orient'], action)
        return action, True

    # ── PUSH ──────────────────────────────────────────────────────────────
    elif phase == P_PUSH:
        if stuck:
            mem['consec_stuck_push'] += 1
            action = R22 if mem['consec_stuck_push'] % 4 < 2 else FW
        else:
            mem['consec_stuck_push'] = 0; action = FW
        mem['orient'] = update_orient(mem['orient'], action)
        return action, False

    mem['orient'] = update_orient(mem['orient'], FW)
    return FW, False

def get_difficulty(ep):
    lvl = DIFFICULTY_SCHEDULE[0][1]
    for s, l in DIFFICULTY_SCHEDULE:
        if ep >= s: lvl = l
    return lvl

def get_wall_prob(ep):
    p = WALL_SCHEDULE[0][1]
    for s, wp in WALL_SCHEDULE:
        if ep >= s: p = wp
    return p

def train():
    print(f"\n{'='*65}")
    print(f"  OBELIX Map-Based Agent | {TOTAL_EPISODES} episodes")
    print(f"  Phases: INIT→SEEK_WALL→ALIGN→SEARCH→FIND_GAP→CROSS→APPROACH→PUSH")
    print(f"  Q-table trained on APPROACH phase only (8-bit compact state)")
    print(f"{'='*65}\n")

    loaded = False
    for path in [SAVE_LATEST, SAVE_BEST_SR]:
        if os.path.exists(path):
            raw = np.load(path)
            if raw.shape[0] == N_STATES:
                q_table = raw
                print(f"Loaded Q-table from {path}")
                loaded = True
                break
    if not loaded:
        q_table = init_approach_qtable()
        print("Fresh Q-table (approach phase, 8-bit state)")

    temperature        = TEMP_START
    success_window     = deque(maxlen=50)
    nowall_success_win = deque(maxlen=50)
    rewards_log        = []
    best_sr = best_sr_nw = 0.0
    log_data = []

    all_phases  = [P_INIT,P_SEEK_WALL,P_ALIGN,P_SEARCH,
                   P_FIND_GAP,P_CROSS_GAP,P_APPROACH,P_PUSH]
    phase_steps = {p: 0 for p in all_phases}
    t0 = time.time()

    for ep in range(TOTAL_EPISODES):
        difficulty = get_difficulty(ep)
        wall_prob  = get_wall_prob(ep)
        use_wall   = random.random() < wall_prob
        max_steps  = MAX_STEPS_WALL if use_wall else MAX_STEPS_NOWALL

        env = OBELIX(scaling_factor=3, difficulty=difficulty,
                     max_steps=max_steps, wall_obstacles=use_wall, seed=None)

        obs      = np.asarray(env.sensor_feedback, dtype=np.float32)
        mem      = make_mem()
        traces   = {}
        prev_obs = obs.copy()
        ep_r     = 0.0
        hist     = deque(maxlen=15)
        burst    = 0
        success  = False

        for step in range(max_steps):
            if burst == 0 and len(hist) >= 8:
                if sum(1 for a in list(hist)[-8:] if a in ROTATIONS) >= 7:
                    burst = 6

            action_idx, is_qtable = fsm_step(
                mem, obs, q_table, temperature, hist, burst, step)

            if is_qtable and burst > 0:
                burst -= 1

            hist.append(action_idx)
            phase_steps[mem['phase']] = phase_steps.get(mem['phase'], 0) + 1

            result   = env.step(ACTIONS[action_idx], render=False)
            next_obs = np.asarray(result[0], dtype=np.float32)
            raw_r    = float(result[1])
            done     = bool(result[2])

            update_mem_approach(mem, next_obs, raw_r)

            if is_qtable:
                any_s = np.any(obs[:17] > 0.5)
                if any_s or mem['pushing']:
                    shaped = shape_approach_reward(raw_r, next_obs, mem, action_idx)
                    state  = encode_approach(obs, mem)
                    ns     = encode_approach(next_obs, mem)
                    td_err = (shaped + GAMMA * float(np.max(q_table[ns]))
                              * (1. - float(done))) - q_table[state, action_idx]
                    update_with_traces(q_table, traces, state, action_idx, td_err)
            else:
                decay_traces(traces)

            ep_r    += raw_r
            prev_obs = next_obs.copy()
            obs      = next_obs

            if done:
                if ep_r > 1800: success = True
                break

        temperature = max(TEMP_END, temperature * TEMP_DECAY)
        rewards_log.append(ep_r)
        success_window.append(int(success))
        if not use_wall:
            nowall_success_win.append(int(success))

        rolling_sr    = float(np.mean(success_window))
        rolling_sr_nw = (float(np.mean(nowall_success_win))
                         if len(nowall_success_win) >= 10 else 0.0)

        if rolling_sr > best_sr and len(success_window) >= 20:
            best_sr = rolling_sr
            np.save(SAVE_BEST_SR, q_table)
            print(f"  ★ New best SR: {best_sr:.3f} (ep {ep+1})")

        if rolling_sr_nw > best_sr_nw and len(nowall_success_win) >= 10:
            best_sr_nw = rolling_sr_nw
            np.save(SAVE_BEST_SR_NW, q_table)

        if (ep + 1) % 50 == 0:
            np.save(SAVE_LATEST, q_table)
            n   = min(500, len(rewards_log))
            avg = np.mean(rewards_log[-n:])
            ela = time.time() - t0
            eta = ((TOTAL_EPISODES-ep-1) / max((ep+1)/ela, 1e-6)) / 60

            total = sum(phase_steps.values())
            pct   = {p: f"{100*v/max(total,1):.0f}%"
                     for p, v in phase_steps.items()}

            print(f"Ep {ep+1:5d}/{TOTAL_EPISODES} | D:{difficulty} W:{int(use_wall)} | "
                  f"SR:{rolling_sr:.2f}(best:{best_sr:.2f}) NW:{rolling_sr_nw:.2f} | "
                  f"Avg({n}):{avg:8.1f} | T:{temperature:.3f} | ETA:{eta:.0f}m")
            print(f"  Phase%: "
                  f"INIT={pct[P_INIT]} WALL={pct[P_SEEK_WALL]} "
                  f"ALIGN={pct[P_ALIGN]} SRCH={pct[P_SEARCH]} "
                  f"GAP={pct[P_FIND_GAP]} CROSS={pct[P_CROSS_GAP]} "
                  f"APP={pct[P_APPROACH]} PUSH={pct[P_PUSH]}")

            log_data.append({
                'ep':ep+1,'sr':rolling_sr,'best_sr':best_sr,
                'avg':avg,'temp':temperature,
            })
            with open(SAVE_LOG,'w') as f: json.dump(log_data,f,indent=2)
            phase_steps = {p: 0 for p in all_phases}

    np.save(SAVE_LATEST, q_table)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m | Best SR:{best_sr:.3f}")
    print(f"Best weights: {SAVE_BEST_SR}")


if __name__ == '__main__':
    train()