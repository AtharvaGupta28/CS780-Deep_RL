TOTAL_EPISODES  = 2000
FRESH_START     = False

REPO_PATH = '/content/drive/MyDrive/CS780-OBELIX'
SAVE_DIR  = '/content/drive/MyDrive/CS780-OBELIX/drqn_v5'

INPUT_DIM       = 26
HIDDEN_DIM      = 128
SEQ_LEN         = 32
BATCH_SIZE      = 16
BUFFER_CAPACITY = 3000
LR_INIT         = 1e-4
LR_MIN          = 5e-6
LR_DECAY_EVERY  = 400
LR_DECAY_FACTOR = 0.65
GAMMA           = 0.97
EPSILON_START   = 0.35
EPSILON_END     = 0.05
TARGET_UPDATE   = 20
GRAD_CLIP       = 0.5
BLIND_THRESHOLD = 20

MAX_STEPS_NOWALL = 650
MAX_STEPS_WALL   = 550

DIFFICULTY_SCHEDULE = [(0, 0), (2500, 2), (4000, 3)]
WALL_SCHEDULE = [(0, 0.20), (1000, 0.30), (2500, 0.40), (4000, 0.45)]

import os
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_BEST   = f'{SAVE_DIR}/drqn_best.pth'
SAVE_LATEST = f'{SAVE_DIR}/drqn_latest.pth'
SAVE_AGENT  = f'{SAVE_DIR}/weights.pth'
SAVE_LOG    = f'{SAVE_DIR}/drqn_log_v5.json'

import sys, random, time, json, math
from collections import deque
import numpy as np

sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
os.system("pip install -r requirements.txt -q")

import torch
import torch.nn as nn
import torch.optim as optim
from obelix import OBELIX

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
L45, L22, FW, R22, R45 = 0, 1, 2, 3, 4
ROTATIONS = {0, 1, 3, 4}
ACTION_ORIENT_DELTA = [2, 1, 0, -1, -2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class DRQN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.gru   = nn.GRU(64, hidden_dim, batch_first=True)
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, n_actions))

    def forward(self, x, h=None):
        b = x.size(0)
        if h is None:
            h = torch.zeros(1, b, self.hidden_dim, device=x.device)
        e = self.embed(x)
        out, h_new = self.gru(e, h)
        return self.head(out), h_new

class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, episode):
        if len(episode) > 2:
            self.buffer.append(episode)

    def sample(self, batch_size, seq_len):
        batch_size = min(batch_size, len(self.buffer))
        episodes   = random.sample(list(self.buffer), batch_size)
        B_i, B_a, B_r, B_d, B_n, B_m = [], [], [], [], [], []
        for ep in episodes:
            L = len(ep)
            if L <= 2: continue
            start  = random.randint(0, max(0, L - seq_len - 1))
            actual = min(seq_len, L - start - 1)
            inp = np.zeros((seq_len, INPUT_DIM), np.float32)
            act = np.zeros(seq_len, np.int64)
            rew = np.zeros(seq_len, np.float32)
            don = np.zeros(seq_len, np.float32)
            nxt = np.zeros((seq_len, INPUT_DIM), np.float32)
            msk = np.zeros(seq_len, np.float32)
            for i in range(actual):
                idx = start + i
                if idx + 1 >= L: break
                inp[i], act[i], rew[i], don[i] = ep[idx]
                nxt[i] = ep[idx+1][0]
                msk[i] = 1.0
            B_i.append(inp); B_a.append(act); B_r.append(rew)
            B_d.append(don); B_n.append(nxt); B_m.append(msk)
        if not B_i: return None
        return (torch.FloatTensor(np.array(B_i)),
                torch.LongTensor(np.array(B_a)),
                torch.FloatTensor(np.array(B_r)),
                torch.FloatTensor(np.array(B_d)),
                torch.FloatTensor(np.array(B_n)),
                torch.FloatTensor(np.array(B_m)))

    def __len__(self): return len(self.buffer)

_TURN_SEQS = [
    [R45, R45],        # 90 right
    [L45, L45],        # 90 left
    [R45, R45, R45],   # 135 right
    [L45, L45, L45],   # 135 left
    [R45, R45, R45, R45],  # 180
]

class RandomWalkSearch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state    = 'WALKING'
        self.turn_seq = []
        self.turn_idx = 0

    def get_action(self, stuck):
        if stuck and self.state == 'WALKING':
            # Pick random turn sequence
            self.turn_seq = random.choice(_TURN_SEQS)[:]
            self.turn_idx = 0
            self.state    = 'TURNING'

        if self.state == 'TURNING':
            if self.turn_idx < len(self.turn_seq):
                act = self.turn_seq[self.turn_idx]
                self.turn_idx += 1
                return act
            else:
                self.state = 'WALKING'
                return FW

        # WALKING
        return FW

def build_input(obs, prev_action, orient, pushing):
    prev_oh    = np.zeros(5, np.float32)
    prev_oh[prev_action] = 1.0
    orient_rad = orient * (2 * math.pi / 16)
    return np.concatenate([
        np.asarray(obs, np.float32), prev_oh,
        [math.sin(orient_rad), math.cos(orient_rad)],
        [float(pushing)],
    ])


def shape_reward(raw_r, obs, next_obs, action_idx, pushing, was_stuck):
    n = np.asarray(next_obs, np.float32)
    o = np.asarray(obs,      np.float32)
    any_now  = np.any(n[:17] > 0.5)
    any_prev = np.any(o[:17] > 0.5)
    ir_now   = n[16] > 0.5
    stuck    = n[17] > 0.5
    fn       = np.any(n[0:4]  > 0.5)
    ff       = np.any(n[8:12] > 0.5)

    if raw_r >= 1800: return 50.0
    if raw_r >= 90:   return 20.0
    if raw_r <= -150: return -5.0

    r = -0.15
    if any_now and not any_prev: r += 8.0
    if fn:                       r += 2.5
    elif ff:                     r += 1.0
    if ir_now and not pushing:   r += 5.0
    if pushing and not stuck:    r += 3.0
    if pushing and stuck:        r -= 0.5
    if action_idx == FW and any_now and not pushing: r += 1.5
    if action_idx == FW and was_stuck and not stuck: r += 3.0
    if action_idx in ROTATIONS and not any_now:      r -= 0.3
    return float(np.clip(r, -10.0, 50.0))


def get_difficulty(ep, best_sr_nowall):
    lvl = 0
    for start, l in DIFFICULTY_SCHEDULE:
        if ep >= start: lvl = l
    if lvl >= 2 and best_sr_nowall < 0.25: lvl = 0
    return lvl

def get_wall_prob(ep):
    p = 0.0
    for start, wp in WALL_SCHEDULE:
        if ep >= start: p = wp
    return p

online_net = DRQN().to(device)
target_net = DRQN().to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()
optimizer  = optim.Adam(online_net.parameters(), lr=LR_INIT)
buffer     = EpisodeReplayBuffer(BUFFER_CAPACITY)
huber      = nn.SmoothL1Loss(reduction='none')

start_ep       = 0
best_sr        = 0.0
best_sr_nowall = 0.0

for ckpt_path in [
    f'{SAVE_DIR}/drqn_latest.pth',
    f'{SAVE_DIR}/drqn_best.pth',
    '/content/drive/MyDrive/CS780-OBELIX/drqn_v4/drqn_best.pth',
    '/content/drive/MyDrive/CS780-OBELIX/drqn_v4/drqn_latest.pth',
]:
    if os.path.exists(ckpt_path) and not FRESH_START:
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            online_net.load_state_dict(ck['model'])
            target_net.load_state_dict(ck.get('target', ck['model']))
            if 'optimizer' in ck:
                try:
                    optimizer.load_state_dict(ck['optimizer'])
                    for g in optimizer.param_groups: g['lr'] = LR_INIT
                except: pass
            start_ep       = ck.get('episode', 0)
            best_sr        = ck.get('best_sr', 0.0)
            best_sr_nowall = ck.get('best_sr_nowall', 0.0)
            print(f"Resumed {ckpt_path} | ep={start_ep} sr={best_sr:.3f}")
            break
        except Exception as e:
            print(f"Load failed {ckpt_path}: {e}")

success_win  = deque(maxlen=100)
reward_win   = deque(maxlen=100)
wall_win     = deque(maxlen=50)
nowall_win   = deque(maxlen=100)
loss_win     = deque(maxlen=100)
stuck_win    = deque(maxlen=100)
log_data     = []
phase_counts = {'search': 0, 'drqn': 0}
t0           = time.time()

print(f"Training {TOTAL_EPISODES} eps from ep {start_ep}")
print(f"Search: random walk (no lawnmower) — StuckAvg target <5")
print("="*70)

for ep in range(start_ep, start_ep + TOTAL_EPISODES):

    difficulty = get_difficulty(ep, best_sr_nowall)
    wall_prob  = get_wall_prob(ep)
    use_wall   = random.random() < wall_prob
    max_steps  = MAX_STEPS_WALL if use_wall else MAX_STEPS_NOWALL

    progress = min(1.0, (ep - start_ep) / max(TOTAL_EPISODES * 0.70, 1))
    epsilon  = EPSILON_START + (EPSILON_END - EPSILON_START) * progress

    rel_ep = ep - start_ep
    if rel_ep > 0 and rel_ep % LR_DECAY_EVERY == 0:
        for g in optimizer.param_groups:
            g['lr'] = max(g['lr'] * LR_DECAY_FACTOR, LR_MIN)
        print(f"  LR -> {optimizer.param_groups[0]['lr']:.2e}")

    env    = OBELIX(scaling_factor=3, difficulty=difficulty,
                    max_steps=max_steps, wall_obstacles=use_wall, seed=None)
    obs    = np.asarray(env.sensor_feedback, np.float32)
    walker = RandomWalkSearch()

    prev_action     = FW
    orient          = 0
    pushing         = False
    hidden          = None
    episode_data    = []
    ep_reward       = 0.0
    success         = False
    steps_no_sensor = 0
    ep_stuck_steps  = 0
    recent_actions  = deque(maxlen=12)

    for step in range(max_steps):
        any_sensor = np.any(obs[:17] > 0.5)
        stuck      = obs[17] > 0.5
        ir_active  = obs[16] > 0.5

        steps_no_sensor = 0 if any_sensor else steps_no_sensor + 1
        if stuck: ep_stuck_steps += 1
        if ir_active and not pushing: pushing = True

        inp   = build_input(obs, prev_action, orient, pushing)
        inp_t = torch.FloatTensor(inp).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals, hidden = online_net(inp_t, hidden)

        use_drqn   = False
        action_idx = FW

        if pushing:
            if stuck:

                action_idx = walker.get_action(stuck)
            else:
                use_drqn = True
            phase_counts['drqn'] += 1

        elif any_sensor:
            use_drqn = True
            phase_counts['drqn'] += 1
            walker.reset()

        elif steps_no_sensor <= BLIND_THRESHOLD:
            use_drqn = True
            phase_counts['drqn'] += 1

        else:
            action_idx = walker.get_action(stuck)
            phase_counts['search'] += 1

        if use_drqn:
            if random.random() < epsilon:
                action_idx = random.randint(0, 4)
            else:
                action_idx = q_vals[0, 0].argmax().item()
            if (len(recent_actions) >= 10
                    and sum(1 for a in recent_actions if a in ROTATIONS) >= 9
                    and not any_sensor):
                action_idx = FW

        recent_actions.append(action_idx)
        prev_stuck = stuck

        result   = env.step(ACTIONS[action_idx], render=False)
        next_obs = np.asarray(result[0], np.float32)
        raw_r    = float(result[1])
        done     = bool(result[2])

        if use_drqn:
            sr = shape_reward(raw_r, obs, next_obs, action_idx,
                              pushing, prev_stuck)
            episode_data.append((inp.copy(), action_idx, sr, done))

        orient      = (orient + ACTION_ORIENT_DELTA[action_idx]) % 16
        prev_action = action_idx
        ep_reward  += raw_r
        obs         = next_obs
        if any_sensor: walker.reset()

        if done:
            if ep_reward > 1800: success = True
            if episode_data:
                final_inp = build_input(next_obs, action_idx, orient, pushing)
                episode_data.append((final_inp.copy(), 0, 0.0, True))
            break

    if not done and episode_data:
        final_inp = build_input(obs, prev_action, orient, pushing)
        episode_data.append((final_inp.copy(), 0, 0.0, True))

    if episode_data:
        buffer.add(episode_data)

    success_win.append(int(success))
    reward_win.append(ep_reward)
    stuck_win.append(ep_stuck_steps)
    (wall_win if use_wall else nowall_win).append(int(success))

    if len(buffer) >= BATCH_SIZE:
        batch = buffer.sample(BATCH_SIZE, SEQ_LEN)
        if batch is not None:
            inp_b, act_b, rew_b, don_b, nxt_b, msk_b = \
                [x.to(device) for x in batch]
            q_all, _  = online_net(inp_b)
            q_taken   = q_all.gather(2, act_b.unsqueeze(2)).squeeze(2)
            with torch.no_grad():
                q_on, _ = online_net(nxt_b)
                best_a  = q_on.argmax(2)
                q_tg, _ = target_net(nxt_b)
                q_next  = q_tg.gather(2, best_a.unsqueeze(2)).squeeze(2)
                targets = rew_b + GAMMA * q_next * (1.0 - don_b)
            loss = (huber(q_taken, targets) * msk_b).sum() / msk_b.sum().clamp(1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_win.append(loss.item())

    if (ep + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

    if (ep + 1) % 50 == 0:
        sr    = np.mean(success_win)  if success_win        else 0
        sr_w  = np.mean(wall_win)     if len(wall_win)  >= 5 else 0
        sr_nw = np.mean(nowall_win)   if len(nowall_win)>= 5 else 0
        avg_r = np.mean(reward_win)   if reward_win         else 0
        avg_l = np.mean(loss_win)     if loss_win           else 0
        avg_sk= np.mean(stuck_win)    if stuck_win          else 0
        ela   = time.time() - t0
        eta   = ((TOTAL_EPISODES-(ep-start_ep)-1)
                 / max((ep-start_ep+1)/ela, 1e-6) / 60)
        tot   = sum(phase_counts.values()) or 1
        lr_now= optimizer.param_groups[0]['lr']

        print(f"Ep {ep+1:5d} | D:{difficulty} W:{int(use_wall)} | "
              f"SR:{sr:.2f}(nw:{sr_nw:.2f} w:{sr_w:.2f}) best:{best_sr:.2f} | "
              f"R:{avg_r:8.0f} | L:{avg_l:.3f} | "
              f"Stuck:{avg_sk:.1f} | "
              f"e:{epsilon:.3f} lr:{lr_now:.1e} | "
              f"S:{100*phase_counts['search']//tot}% "
              f"D:{100*phase_counts['drqn']//tot}% | "
              f"ETA:{eta:.0f}m")

        if sr > best_sr and len(success_win) >= 20:
            best_sr = sr
            torch.save({'model': online_net.state_dict(),
                        'target': target_net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'episode': ep+1, 'best_sr': best_sr,
                        'best_sr_nowall': best_sr_nowall,
                        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
                        'n_actions': N_ACTIONS}, SAVE_BEST)
            torch.save({'model': online_net.state_dict(),
                        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
                        'n_actions': N_ACTIONS}, SAVE_AGENT)
            print(f"  New best SR: {best_sr:.3f}")

        if sr_nw > best_sr_nowall:
            best_sr_nowall = sr_nw

        torch.save({'model': online_net.state_dict(),
                    'target': target_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode': ep+1, 'best_sr': best_sr,
                    'best_sr_nowall': best_sr_nowall,
                    'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
                    'n_actions': N_ACTIONS}, SAVE_LATEST)

        log_data.append({'ep': ep+1, 'sr': sr, 'sr_wall': sr_w,
                         'sr_nowall': sr_nw, 'best_sr': best_sr,
                         'avg_reward': avg_r, 'loss': avg_l,
                         'avg_stuck': avg_sk, 'epsilon': epsilon,
                         'difficulty': difficulty, 'lr': lr_now})
        with open(SAVE_LOG, 'w') as f:
            json.dump(log_data, f, indent=2)
        phase_counts = {'search': 0, 'drqn': 0}

torch.save({'model': online_net.state_dict(),
            'target': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'episode': start_ep + TOTAL_EPISODES,
            'best_sr': best_sr, 'best_sr_nowall': best_sr_nowall,
            'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
            'n_actions': N_ACTIONS}, SAVE_LATEST)
torch.save({'model': online_net.state_dict(),
            'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM,
            'n_actions': N_ACTIONS}, SAVE_AGENT)

print(f"\n{'='*60}")
print(f"Done in {(time.time()-t0)/60:.1f}m | Best SR: {best_sr:.3f}")
print(f"Weights -> {SAVE_AGENT}")
print(f"{'='*60}")