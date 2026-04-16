import sys, os

REPO_PATH  = '/content/drive/MyDrive/CS780-OBELIX'
SAVE_DIR   = '/content/drive/MyDrive/CS780-OBELIX'

sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
os.system("pip install -r requirements.txt -q")

import random, time, json
from collections import deque

import numpy as np
from obelix import OBELIX

TOTAL_EPISODES     = 8000      
MAX_STEPS_NOWALL   = 600
MAX_STEPS_WALL     = 350

ALPHA              = 0.18      
ALPHA_NEIGHBOUR    = 0.05      
GAMMA              = 0.96
LAMBDA             = 0.85

TEMP_START         = 6.0        
TEMP_END           = 0.4       
TEMP_DECAY         = 0.9993    

CURIOSITY_SCALE    = 4.0       

REWARD_CLIP_LOW    = -30.0     
REWARD_CLIP_HIGH   =  60.0

DIFFICULTY_SCHEDULE = [
    (0,    0),   
    (1500, 2),    
    (3000, 3),    
]

WALL_SCHEDULE = [
    (0,    0.00),
    (1000, 0.15),
    (2000, 0.30),
    (3500, 0.50),
    (5000, 0.65),
]

PSCALE_RAMP_END    = 2500

ANTI_SPIN_WINDOW   = 10        # consecutive steps checked for spin
SPIN_THRESHOLD     = 9         # how many of those must be rotations
BURST_LENGTH       = 10        # forced FW steps after spin detected

MAX_STUCK_STEPS    = 20
VISIBILITY_HORIZON = 25

N_BITS   = 12
N_STATES = 2 ** N_BITS         # 4096 states

HAMMING_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 1, 6, 6, 6, 1, 1, 1], dtype=np.float32
)
HAMMING_THRESHOLD = 2 

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
ROTATIONS = {0, 1, 3, 4} 
FW        = 2

SAVE_BEST_SR    = os.path.join(SAVE_DIR, 'q_best_sr_v3.npy')
SAVE_BEST_SR_NW = os.path.join(SAVE_DIR, 'q_best_sr_nowall_v3.npy')
SAVE_LATEST     = os.path.join(SAVE_DIR, 'q_latest_v3.npy')
SAVE_VISITS     = os.path.join(SAVE_DIR, 'q_visits_v3.npy')
SAVE_LOG        = os.path.join(SAVE_DIR, 'training_log_v3.json')

print("Precomputing Hamming neighbour mask (4096×4096)…")
_t = time.time()
_ALL_BITS = np.array(
    [[(s >> b) & 1 for b in range(N_BITS)] for s in range(N_STATES)],
    dtype=np.float32
)
_NEIGHBOUR_MASK = np.zeros((N_STATES, N_STATES), dtype=bool)
for s in range(N_STATES):
    d = np.abs(_ALL_BITS - _ALL_BITS[s]) @ HAMMING_WEIGHTS
    _NEIGHBOUR_MASK[s] = (d <= HAMMING_THRESHOLD) & (np.arange(N_STATES) != s)
avg_nb = _NEIGHBOUR_MASK.sum(1).mean()
print(f"  Done in {time.time()-_t:.1f}s | avg neighbours per state: {avg_nb:.1f}")


def get_difficulty(ep: int) -> int:
    level = DIFFICULTY_SCHEDULE[0][1]
    for (start, lvl) in DIFFICULTY_SCHEDULE:
        if ep >= start:
            level = lvl
    return level


def get_wall_prob(ep: int) -> float:
    prob = WALL_SCHEDULE[0][1]
    for (start, p) in WALL_SCHEDULE:
        if ep >= start:
            prob = p
    return prob


def get_pscale(ep: int) -> float:
    return min(1.0, 0.2 + 0.8 * (ep / PSCALE_RAMP_END))


def get_temperature(ep: int, temp: float) -> float:
    """Decay temperature each episode, floored at TEMP_END."""
    return max(TEMP_END, temp * TEMP_DECAY)


def encode(obs: np.ndarray, mem: dict) -> int:
    fn = int(np.any(obs[0:4]   > 0.5))
    ln = int(np.any(obs[4:6]   > 0.5))
    rn = int(np.any(obs[6:8]   > 0.5))
    ff = int(np.any(obs[8:12]  > 0.5))
    lf = int(np.any(obs[12:14] > 0.5))
    rf = int(np.any(obs[14:16] > 0.5))
    ir = int(obs[16] > 0.5)
    sk = int(obs[17] > 0.5)
    ps = int(mem['pushing'])
    rv = int(mem['recently_visible'])
    bl = int(mem['steps_since_visible'] > 10)
    gs = int(mem['in_gap_search'])
    return (fn | (ln<<1) | (rn<<2) | (ff<<3) | (lf<<4) | (rf<<5) |
            (ir<<6) | (sk<<7) | (ps<<8) | (rv<<9) | (bl<<10) | (gs<<11))


def make_mem() -> dict:
    return {
        # task flags
        'pushing':              False,
        'recently_visible':     False,
        'steps_since_visible':  0,
        # wall/gap
        'in_gap_search':        False,
        'consec_stuck':         0,
        'gap_seq':              [],
        'gap_dir':              0,
        'gap_scan_len':         4,
        # drift tracking (for blind exploration bias)
        'last_seen_left':       False,
        'last_seen_right':      False,
        'drift_dir':            None,
        # penalty type decoded last step
        'last_penalty_type':    None,
        # eligibility traces
        'traces':               {},
        # spin burst
        'burst_remaining':      0,
    }


def update_mem(mem: dict, obs: np.ndarray, prev_obs: np.ndarray,
               raw_reward: float) -> None:
    mem['consec_stuck'] = mem['consec_stuck'] + 1 if obs[17] > 0.5 else 0

    # Attachment detection
    if raw_reward >= 90:
        mem['pushing'] = True

    # Penalty type discrimination (Fix 1)
    if raw_reward <= -150:
        if mem['pushing']:
            mem['last_penalty_type'] = 'box_into_wall'
        elif obs[17] > 0.5:
            mem['last_penalty_type'] = 'robot_wall'
        else:
            mem['last_penalty_type'] = 'boundary'
    else:
        mem['last_penalty_type'] = None

    # Visibility tracking
    any_s = np.any(obs[:17] > 0.5)
    lnow  = np.any(obs[4:6] > 0.5) or np.any(obs[12:14] > 0.5)
    rnow  = np.any(obs[6:8] > 0.5) or np.any(obs[14:16] > 0.5)

    if any_s:
        mem['steps_since_visible'] = 0
        mem['recently_visible']    = True
        if mem['last_seen_left']  and rnow and not lnow:
            mem['drift_dir'] = 'right'
        elif mem['last_seen_right'] and lnow and not rnow:
            mem['drift_dir'] = 'left'
        mem['last_seen_left']  = lnow
        mem['last_seen_right'] = rnow
    else:
        mem['steps_since_visible'] += 1
        if mem['steps_since_visible'] > VISIBILITY_HORIZON:
            mem['recently_visible'] = False
            mem['drift_dir']        = None

def _build_gap_seq(gap_dir: int, scan_len: int) -> list:
    """
    Build a heuristic action sequence that sweeps along the wall
    looking for the gap, then tries the other direction if needed.
    gap_dir=0 → start left, gap_dir=1 → start right.
    """
    L90 = [0, 0]    # two L45s = 90° left
    R90 = [4, 4]    # two R45s = 90° right
    T180 = [0, 0, 0, 0]  # 4×L45 = 180°

    par  = L90 if gap_dir == 0 else R90   # parallel to wall
    back = R90 if gap_dir == 0 else L90   # face wall again
    b2   = L90 if gap_dir == 0 else R90
    p2   = R90 if gap_dir == 0 else L90

    seq = list(par)
    for _ in range(3):
        seq += [FW] * scan_len + back + [FW, FW, FW] + list(par)
    seq += T180
    for _ in range(3):
        seq += [FW] * (scan_len * 2) + b2 + [FW, FW, FW] + list(p2)
    return seq


def init_gap(mem: dict) -> None:
    mem['in_gap_search'] = True
    mem['gap_seq']       = _build_gap_seq(mem['gap_dir'], mem['gap_scan_len'])
    mem['traces']        = {}   # reset traces — FSM actions are heuristic


def get_gap_action(mem: dict) -> int:
    if not mem['gap_seq']:
        mem['gap_scan_len'] = min(mem['gap_scan_len'] + 2, 18)
        mem['gap_dir']      = 1 - mem['gap_dir']
        mem['gap_seq']      = _build_gap_seq(mem['gap_dir'], mem['gap_scan_len'])
    return mem['gap_seq'].pop(0)


def exit_gap(mem: dict) -> None:
    mem['in_gap_search'] = False
    mem['gap_seq']       = []
    mem['gap_scan_len']  = 4
    mem['gap_dir']       = 0
    mem['traces']        = {}   # fresh traces after heuristic sequence

def shape_reward(raw_r: float, obs: np.ndarray, prev_obs: np.ndarray,
                 mem: dict, action_idx: int, action_history: deque,
                 pscale: float, visit_counts: np.ndarray, state: int) -> float:

    r         = float(raw_r)
    any_now   = np.any(obs[:17]  > 0.5)
    any_prev  = np.any(prev_obs[:17] > 0.5)
    ir        = obs[16] > 0.5
    stuck     = obs[17] > 0.5
    was_stuck = prev_obs[17] > 0.5
    fn        = np.any(obs[0:4]  > 0.5)
    ff        = np.any(obs[8:12] > 0.5)

    pt = mem['last_penalty_type']
    if pt == 'robot_wall':
        r = r * 0.05            # drastically reduce raw -200
        r -= 4.0 * pscale
    elif pt == 'box_into_wall':
        r = r * 0.10
        r -= 12.0 * pscale
    elif pt == 'boundary':
        r = 0.0
        r -= 2.0

    if any_now and not any_prev: r += 6.0    # first contact with box region
    if fn:                       r += 2.5    # front near
    elif ff:                     r += 1.0    # front far

    if ir and not mem['pushing']: r += 4.0

    if mem['pushing'] and not stuck: r += 1.0

    if mem['recently_visible'] and not any_now: r -= 2.0

    if action_idx == FW and any_now:        r += 2.5   # approach when visible
    if action_idx == FW and not any_now and not stuck: r += 0.8   # explore when blind

    if action_idx == FW and was_stuck and not stuck:
        r += 20.0              # found the gap and moved through it

    if action_idx in ROTATIONS and len(action_history) >= 3:
        if sum(1 for a in list(action_history)[-3:] if a in ROTATIONS) >= 3:
            r -= 2.0

    if any_now != any_prev: r += 1.5

    visit_n = max(1, int(visit_counts[state]))
    r += CURIOSITY_SCALE / np.sqrt(visit_n)

    if raw_r < 90:   # not an attachment or terminal reward
        r = float(np.clip(r, REWARD_CLIP_LOW, REWARD_CLIP_HIGH))

    return r

TRACE_THRESHOLD = 0.005


def update_with_traces(q_table: np.ndarray, mem: dict,
                       state: int, action_idx: int, td_error: float) -> None:
    traces = mem['traces']
    decay  = GAMMA * LAMBDA

    # Decay all existing traces; prune small ones
    dead = [k for k, v in traces.items() if abs(v * decay) < TRACE_THRESHOLD]
    for k in dead:
        del traces[k]
    for k in traces:
        traces[k] *= decay

    # Accumulate current (state, action) — replacing-traces strategy
    key = (state, action_idx)
    traces[key] = max(traces.get(key, 0.0) + 1.0, 1.0)

    # Apply update to all traced (state, action) pairs + Hamming neighbours
    for (s, a), trace_val in traces.items():
        delta = ALPHA * td_error * trace_val
        q_table[s, a] += delta
        q_table[_NEIGHBOUR_MASK[s], a] += ALPHA_NEIGHBOUR * td_error * trace_val


def decay_traces_only(mem: dict) -> None:
    """Call during heuristic (gap FSM) steps — decay but don't update Q."""
    traces = mem['traces']
    decay  = GAMMA * LAMBDA
    dead   = [k for k, v in traces.items() if abs(v * decay) < TRACE_THRESHOLD]
    for k in dead:
        del traces[k]
    for k in traces:
        traces[k] *= decay


def init_qtable() -> np.ndarray:
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for s in range(N_STATES):
        fn = (s >> 0) & 1;  ln = (s >> 1) & 1;  rn = (s >> 2) & 1
        ff = (s >> 3) & 1;  lf = (s >> 4) & 1;  rf = (s >> 5) & 1
        ir = (s >> 6) & 1;  sk = (s >> 7) & 1
        ps = (s >> 8) & 1

        if sk and not ps:
            q[s, 0] = 1.0; q[s, 4] = 1.0          # stuck → try turns
        elif ir or ps:
            q[s, FW] = 5.0                          # IR/pushing → go forward
            q[s, 1] = -0.5; q[s, 3] = -0.5
            q[s, 0] = -1.5; q[s, 4] = -1.5
        elif fn:
            q[s, FW] = 3.0
        elif ff:
            q[s, FW] = 2.0
        elif ln and not rn:
            q[s, 1] = 2.0;  q[s, FW] = 0.8         # box left → turn L22
        elif rn and not ln:
            q[s, 3] = 2.0;  q[s, FW] = 0.8
        elif lf and not rf:
            q[s, 1] = 1.5;  q[s, FW] = 0.5
        elif rf and not lf:
            q[s, 3] = 1.5;  q[s, FW] = 0.5
        else:
            # No sensors: strongly bias FW to explore
            q[s, FW] = 2.0
            q[s, 1]  = 0.3;  q[s, 3] = 0.3
            q[s, 0]  = 0.1;  q[s, 4] = 0.1
    return q

def select_action(state: int, q_table: np.ndarray, temperature: float,
                  obs: np.ndarray, mem: dict,
                  action_history: deque, burst: int) -> int:

    if burst > 0:
        return FW

    any_s = np.any(obs[:17] > 0.5)

    if len(action_history) >= ANTI_SPIN_WINDOW:
        recent = list(action_history)[-ANTI_SPIN_WINDOW:]
        if sum(1 for a in recent if a in ROTATIONS) >= SPIN_THRESHOLD:
            if random.random() < 0.90:
                return FW

    q_vals = q_table[state].copy()

    # Penalise rotations when sensors silent (soft bias, not hard block)
    if not any_s:
        for a in ROTATIONS:
            q_vals[a] -= 1.5
        # Drift bias: if we last saw the box on one side, lean that way
        if mem['drift_dir'] == 'right':
            q_vals[3] += 0.5; q_vals[4] += 0.3   # R22, R45
        elif mem['drift_dir'] == 'left':
            q_vals[1] += 0.5; q_vals[0] += 0.3   # L22, L45

    # Boltzmann sampling
    q_shifted = q_vals - np.max(q_vals)           # numerical stability
    probs = np.exp(q_shifted / max(temperature, 0.05))
    probs /= probs.sum()
    return int(np.random.choice(N_ACTIONS, p=probs))

def check_spin(action_history: deque) -> bool:
    """Returns True if the last ANTI_SPIN_WINDOW actions are mostly rotations."""
    if len(action_history) < ANTI_SPIN_WINDOW:
        return False
    recent = list(action_history)[-ANTI_SPIN_WINDOW:]
    return sum(1 for a in recent if a in ROTATIONS) >= SPIN_THRESHOLD

def train():
    print(f"\n{'='*65}")
    print(f"  OBELIX Phase 3 | FINAL OPTIMISED | {TOTAL_EPISODES} episodes")
    print(f"  Boltzmann exploration | Curiosity | Q(λ={LAMBDA})")
    print(f"  Saves by: ROLLING SUCCESS RATE (last 100 eps)")
    print(f"{'='*65}\n")

    # ── Load or init Q-table ──
    if os.path.exists(SAVE_LATEST):
        q_table = np.load(SAVE_LATEST)
        print(f"Resumed from {SAVE_LATEST}")
    else:
        q_table = init_qtable()
        print("Initialised fresh Q-table with heuristic warm-start")

    visit_counts = (np.load(SAVE_VISITS)
                    if os.path.exists(SAVE_VISITS)
                    else np.zeros(N_STATES, dtype=np.int64))

    temperature      = TEMP_START
    success_window   = deque(maxlen=100)   # rolling 100-ep success rate
    rewards_log      = []
    wall_success_win = deque(maxlen=100)
    nowall_success_win = deque(maxlen=100)

    best_sr       = 0.0   # best rolling success rate (any)
    best_sr_wall  = 0.0
    best_sr_nw    = 0.0

    log_data = []   # for JSON export
    t0       = time.time()
    ep_start = 0    # for ETA

    for ep in range(TOTAL_EPISODES):
        difficulty = get_difficulty(ep)
        wall_prob  = get_wall_prob(ep)
        pscale     = get_pscale(ep)
        use_wall   = random.random() < wall_prob
        max_steps  = MAX_STEPS_WALL if use_wall else MAX_STEPS_NOWALL

        env = OBELIX(
            scaling_factor=3,
            difficulty=difficulty,
            max_steps=max_steps,
            wall_obstacles=use_wall,
            seed=None
        )

        obs      = np.asarray(env.sensor_feedback, dtype=np.float32)
        mem      = make_mem()
        prev_obs = obs.copy()
        ep_r     = 0.0
        hist     = deque(maxlen=ANTI_SPIN_WINDOW + 5)
        burst    = 0          
        success  = False

        for step in range(max_steps):
            state = encode(obs, mem) % N_STATES
            visit_counts[state] += 1
            stuck = obs[17] > 0.5

            if burst == 0 and check_spin(hist):
                burst = BURST_LENGTH

            is_heuristic = False

            if mem['in_gap_search']:
                action_idx   = get_gap_action(mem)
                is_heuristic = True
                # Exit FSM if bot escaped wall
                if not stuck and step > 0 and prev_obs[17] > 0.5:
                    exit_gap(mem)
                    is_heuristic = False
                    burst        = 0
                    action_idx   = select_action(state, q_table, temperature,
                                                 obs, mem, hist, burst)

            elif stuck and not mem['pushing']:
                if not mem['in_gap_search']:
                    init_gap(mem)
                action_idx   = get_gap_action(mem)
                is_heuristic = True
                if mem['consec_stuck'] >= MAX_STUCK_STEPS:
                    mem['gap_dir']      = 1 - mem['gap_dir']
                    mem['gap_scan_len'] = min(mem['gap_scan_len'] + 2, 18)
                    mem['consec_stuck'] = 0
                    init_gap(mem)

            else:
                action_idx = select_action(state, q_table, temperature,
                                           obs, mem, hist, burst)
                if burst > 0:
                    burst -= 1

            hist.append(action_idx)

            result   = env.step(ACTIONS[action_idx], render=False)
            next_obs = np.asarray(result[0], dtype=np.float32)
            raw_r    = float(result[1])
            done     = bool(result[2])

            update_mem(mem, next_obs, prev_obs, raw_r)
            ns     = encode(next_obs, mem) % N_STATES
            reward = shape_reward(raw_r, next_obs, prev_obs, mem,
                                  action_idx, hist, pscale,
                                  visit_counts, state)

            if not is_heuristic:
                best_next = float(np.max(q_table[ns]))
                td_target = reward + GAMMA * best_next * (1.0 - float(done))
                td_error  = td_target - q_table[state, action_idx]
                update_with_traces(q_table, mem, state, action_idx, td_error)
            else:
                decay_traces_only(mem)

            ep_r    += raw_r
            prev_obs = next_obs.copy()
            obs      = next_obs

            if done:
                if raw_r >= 1800 or ep_r > 1800:
                    success = True
                break

        temperature = get_temperature(ep, temperature)
        rewards_log.append(ep_r)
        success_window.append(int(success))
        (wall_success_win if use_wall else nowall_success_win).append(int(success))

        rolling_sr    = float(np.mean(success_window))
        rolling_sr_w  = float(np.mean(wall_success_win))   if len(wall_success_win)  >= 10 else 0.0
        rolling_sr_nw = float(np.mean(nowall_success_win)) if len(nowall_success_win) >= 10 else 0.0

        if rolling_sr > best_sr and len(success_window) >= 20:
            best_sr = rolling_sr
            np.save(SAVE_BEST_SR, q_table)
            np.save(SAVE_VISITS, visit_counts)

        if rolling_sr_nw > best_sr_nw and len(nowall_success_win) >= 20:
            best_sr_nw = rolling_sr_nw
            np.save(SAVE_BEST_SR_NW, q_table)

        if (ep + 1) % 100 == 0:
            np.save(SAVE_LATEST, q_table)   # always keep latest checkpoint

            n   = min(500, len(rewards_log))
            avg = np.mean(rewards_log[-n:])
            ela = time.time() - t0
            eps_per_sec = (ep + 1) / max(ela, 1e-6)
            eta_min = (TOTAL_EPISODES - ep - 1) / max(eps_per_sec, 1e-6) / 60

            print(
                f"Ep {ep+1:5d}/{TOTAL_EPISODES} | "
                f"Diff:{difficulty} Wall:{int(use_wall)} | "
                f"SR:{rolling_sr:.2f}(best:{best_sr:.2f}) "
                f"W-SR:{rolling_sr_w:.2f} NW-SR:{rolling_sr_nw:.2f} | "
                f"Avg({n}):{avg:7.1f} | "
                f"T:{temperature:.3f} pscl:{pscale:.2f} | "
                f"ETA:{eta_min:.0f}m"
            )

            log_data.append({
                'ep':            ep + 1,
                'difficulty':    difficulty,
                'use_wall':      int(use_wall),
                'rolling_sr':    rolling_sr,
                'best_sr':       best_sr,
                'avg_reward':    avg,
                'temperature':   temperature,
            })
            with open(SAVE_LOG, 'w') as f:
                json.dump(log_data, f, indent=2)

    np.save(SAVE_LATEST, q_table)
    np.save(SAVE_VISITS, visit_counts)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*65}")
    print(f"  Training complete in {elapsed:.1f}m")
    print(f"  Best rolling SR: {best_sr:.3f}")
    print(f"  Best NW-SR:      {best_sr_nw:.3f}")
    print(f"  Weights → {SAVE_BEST_SR}")
    print(f"{'='*65}")

if __name__ == '__main__':
    train()
