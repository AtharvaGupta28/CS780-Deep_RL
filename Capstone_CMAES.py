
N_GENERATIONS    = 80
POP_SIZE         = 20
EVALS_PER_IND    = 5
SR_EVAL_EPISODES = 3
SAVE_EVERY       = 10
MAX_STEPS        = 1000

CURRICULUM = [
    (0,  [(0, False, 1)]),                                            
    (15, [(0, False, 2), (0, True, 1)]),                              
    (30, [(0, False, 2), (0, True, 1), (2, False, 2), (2, True, 1)]), 
    (50, [(0, False, 2), (0, True, 1), (2, False, 2), (2, True, 1),
          (3, False, 2), (3, True, 1)]),                            
]

REPO_PATH = '/content/drive/MyDrive/CS780-OBELIX'
SAVE_DIR  = '/content/drive/MyDrive/CS780-OBELIX/test_phase'

import sys, os, random, time, json, math
import numpy as np

sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
os.system("pip install -r requirements.txt -q")

try:
    import cma
except ImportError:
    os.system("pip install cma -q")
    import cma

from obelix import OBELIX

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
L45, L22, FW, R22, R45 = 0, 1, 2, 3, 4

ACTION_DTHETA = [
    math.radians(45), math.radians(22.5), 0.0,
    math.radians(-22.5), math.radians(-45),
]

ARENA_SIZE    = 500.0
STEP_SIZE     = 5.0
SONAR_NEAR    = 45.0
SONAR_FAR     = 90.0
BOUNDARY_PAD  = 20.0
WALL_X_APPROX = 250.0

BELIEF_DIM = 23
HIDDEN     = 32
N_PARAMS   = BELIEF_DIM * HIDDEN + HIDDEN + HIDDEN * N_ACTIONS + N_ACTIONS

SAVE_LATEST   = os.path.join(SAVE_DIR, 'cmaes_latest.npz')
SAVE_BEST_SR  = os.path.join(SAVE_DIR, 'cmaes_best_sr.npz')
SAVE_BEST_REW = os.path.join(SAVE_DIR, 'cmaes_best_rew.npz')
SAVE_AGENT    = os.path.join(SAVE_DIR, 'cmaes_weights.npz')
SAVE_LOG      = os.path.join(SAVE_DIR, 'cmaes_log.json')

class BeliefState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rx = ARENA_SIZE / 2.0
        self.ry = ARENA_SIZE / 2.0
        self.theta = 0.0
        self.box_x = ARENA_SIZE / 2.0
        self.box_y = ARENA_SIZE / 2.0
        self.box_confidence = 0.0
        self.pushing = False
        self.steps_no_sensor = 0
        self.wall_detected = False
        self.wall_x = WALL_X_APPROX
        self.prev_action = FW

    def update(self, obs, action_idx, raw_reward):
        obs = np.asarray(obs, dtype=np.float32)
        self.theta = (self.theta + ACTION_DTHETA[action_idx]) % (2 * math.pi)

        stuck = obs[17] > 0.5
        if action_idx == FW and not stuck:
            self.rx += STEP_SIZE * math.cos(self.theta)
            self.ry -= STEP_SIZE * math.sin(self.theta)

        if stuck and not self.pushing:
            self.rx = np.clip(self.rx, BOUNDARY_PAD, ARENA_SIZE - BOUNDARY_PAD)
            self.ry = np.clip(self.ry, BOUNDARY_PAD, ARENA_SIZE - BOUNDARY_PAD)
            if abs(self.rx - WALL_X_APPROX) < 60:
                self.wall_detected = True
                self.wall_x = self.rx

        if raw_reward >= 90:
            self.pushing = True

        any_sensor = np.any(obs[:17] > 0.5)
        ir = obs[16] > 0.5

        if any_sensor:
            self.steps_no_sensor = 0
            fn = np.any(obs[0:4] > 0.5)
            ln = np.any(obs[4:6] > 0.5)
            rn = np.any(obs[6:8] > 0.5)
            ff = np.any(obs[8:12] > 0.5)
            lf = np.any(obs[12:14] > 0.5)
            rf = np.any(obs[14:16] > 0.5)

            bearing = 0.0
            if ln or lf: bearing += math.radians(45)
            if rn or rf: bearing -= math.radians(45)

            if ir:               dist = 5.0
            elif fn or ln or rn: dist = SONAR_NEAR * 0.7
            elif ff or lf or rf: dist = SONAR_FAR * 0.7
            else:                dist = SONAR_FAR

            wb = self.theta + bearing
            self.box_x = self.rx + dist * math.cos(wb)
            self.box_y = self.ry - dist * math.sin(wb)
            self.box_confidence = 1.0
        else:
            self.steps_no_sensor += 1
            self.box_confidence *= 0.95

        if self.pushing:
            self.box_x = self.rx + 10 * math.cos(self.theta)
            self.box_y = self.ry - 10 * math.sin(self.theta)
            self.box_confidence = 1.0

        self.prev_action = action_idx

    def get_features(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        dx = self.box_x - self.rx
        dy = self.box_y - self.ry
        d = math.sqrt(dx*dx + dy*dy + 1e-6)
        a = math.atan2(-dy, dx) - self.theta
        db = min(self.rx, self.ry, ARENA_SIZE - self.rx, ARENA_SIZE - self.ry)
        prev_oh = [0.0] * 5
        prev_oh[self.prev_action] = 1.0

        return np.array([
            (self.rx / ARENA_SIZE) * 2 - 1,
            (self.ry / ARENA_SIZE) * 2 - 1,
            math.sin(self.theta), math.cos(self.theta),
            (self.box_x / ARENA_SIZE) * 2 - 1,
            (self.box_y / ARENA_SIZE) * 2 - 1,
            self.box_confidence,
            min(d / ARENA_SIZE, 1.0),
            math.sin(a), math.cos(a),
            min(self.steps_no_sensor / 200.0, 1.0),
            float(self.pushing),
            float(obs[17] > 0.5),
            float(obs[16] > 0.5),
            float(np.any(obs[:17] > 0.5)),
            min(db / (ARENA_SIZE / 2), 1.0),
            float(self.wall_detected),
            (self.wall_x / ARENA_SIZE) * 2 - 1,
        ] + prev_oh, dtype=np.float32)

class PolicyNet:
    def __init__(self):
        self.set_params(np.zeros(N_PARAMS))

    def set_params(self, p):
        i = 0
        n = BELIEF_DIM * HIDDEN
        self.w1 = p[i:i+n].reshape(BELIEF_DIM, HIDDEN); i += n
        n = HIDDEN
        self.b1 = p[i:i+n]; i += n
        n = HIDDEN * N_ACTIONS
        self.w2 = p[i:i+n].reshape(HIDDEN, N_ACTIONS); i += n
        n = N_ACTIONS
        self.b2 = p[i:i+n]; i += n

    def forward(self, f):
        x = np.tanh(f @ self.w1 + self.b1)
        return int(np.argmax(x @ self.w2 + self.b2))

def get_configs_for_gen(gen):
    """Return list of (diff, wall, weight) for current generation."""
    configs = CURRICULUM[0][1]
    for start_gen, cfgs in CURRICULUM:
        if gen >= start_gen:
            configs = cfgs
    return configs

def pick_config(configs):
    weights = [c[2] for c in configs]
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = np.random.choice(len(configs), p=probs)
    return configs[idx][0], configs[idx][1]

def run_episode(params, difficulty, use_wall):
    policy = PolicyNet()
    policy.set_params(params)
    belief = BeliefState()

    env = OBELIX(scaling_factor=3, difficulty=difficulty,
                 max_steps=MAX_STEPS, wall_obstacles=use_wall, seed=None)
    obs = np.asarray(env.sensor_feedback, dtype=np.float32)
    cum_r = 0.0

    for step in range(MAX_STEPS):
        action_idx = policy.forward(belief.get_features(obs))
        result     = env.step(ACTIONS[action_idx], render=False)
        next_obs   = np.asarray(result[0], dtype=np.float32)
        raw_r      = float(result[1])
        done       = bool(result[2])
        belief.update(next_obs, action_idx, raw_r)
        cum_r += raw_r
        obs = next_obs
        if done:
            break

    success = cum_r > 1800
    return cum_r, success

def eval_fitness(params, gen):
    """
    FITNESS = -(successes * 10000 + mean_reward)
    Successes dominate. Reward breaks ties.
    CMA-ES minimizes, so negate.
    """
    configs = get_configs_for_gen(gen)
    rewards = []
    successes = 0

    for _ in range(EVALS_PER_IND):
        diff, wall = pick_config(configs)
        r, s = run_episode(params, diff, wall)
        rewards.append(r)
        if s:
            successes += 1

    mean_r = np.mean(rewards)
    # Normalize reward to [0, 1] range for clean combination
    # Reward typically ranges from -50000 to +2000
    norm_r = (mean_r + 50000) / 52000  
    norm_r = np.clip(norm_r, 0, 1)

    fitness = successes + norm_r   # successes in integers, reward in [0,1]
    return -fitness                # CMA-ES minimizes

def eval_sr(params, gen):
    configs = get_configs_for_gen(gen)
    succ = 0
    total_r = 0.0
    for _ in range(SR_EVAL_EPISODES):
        diff, wall = pick_config(configs)
        r, s = run_episode(params, diff, wall)
        total_r += r
        if s: succ += 1
    return succ / SR_EVAL_EPISODES, total_r / SR_EVAL_EPISODES

eps_per_gen = POP_SIZE * EVALS_PER_IND + SR_EVAL_EPISODES
total_eps   = N_GENERATIONS * eps_per_gen
est_hours   = total_eps * 15 / 3600

print(f"\n{'='*70}")
print(f"  OBELIX CMA-ES — SUCCESS-COUNT FITNESS")
print(f"  fitness = -(successes + normalized_reward)")
print(f"  {N_GENERATIONS} gen × {eps_per_gen} eps/gen = {total_eps} episodes")
print(f"  ~{est_hours:.0f}h estimated | Pop={POP_SIZE} | {EVALS_PER_IND} evals/ind")
print(f"  {N_PARAMS} params | Curriculum: diff0 → diff0+2 → all")
print(f"{'='*70}\n")

# Resume or fresh
x0     = np.zeros(N_PARAMS)
sigma0 = 0.5

if os.path.exists(SAVE_LATEST):
    try:
        data = np.load(SAVE_LATEST, allow_pickle=True)
        if data['best_params'].shape[0] == N_PARAMS:
            x0 = data['best_params']
            if 'sigma' in data:
                sigma0 = max(float(data['sigma']), 0.15)
            print(f"Resumed from checkpoint, σ={sigma0:.3f}")
        else:
            print(f"Param mismatch, starting fresh")
    except:
        print("Checkpoint load failed, starting fresh")
else:
    print("Starting fresh")

opts = cma.CMAOptions()
opts['popsize']   = POP_SIZE
opts['maxiter']   = N_GENERATIONS
opts['verb_disp'] = 0
opts['seed']      = 42
opts['tolx']      = 1e-12
opts['tolfun']    = 1e-12

es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

best_rew_ever    = -1e9
best_sr_ever     = 0.0
best_params_rew  = x0.copy()
best_params_sr   = x0.copy()
log_data         = []
t0 = time.time()

gen = 0
while not es.stop() and gen < N_GENERATIONS:
    gen_t0    = time.time()
    solutions = es.ask()

    configs = get_configs_for_gen(gen)
    diffs_in_play = sorted(set(c[0] for c in configs))
    walls_in_play = any(c[1] for c in configs)
    stage_str = f"D:{diffs_in_play} W:{'Y' if walls_in_play else 'N'}"

    fitnesses   = []
    gen_rewards = []
    gen_succs   = []

    for params in solutions:
        f = eval_fitness(params, gen)
        fitnesses.append(f)
        # Decode for logging
        raw_fitness = -f
        s_count = int(raw_fitness)
        gen_rewards.append(raw_fitness)
        gen_succs.append(s_count)

    es.tell(solutions, fitnesses)

    gen_best_idx = np.argmin(fitnesses)  # best = lowest (most negative)
    gen_best_p   = solutions[gen_best_idx]
    gen_best_r   = gen_rewards[gen_best_idx]
    gen_mean_r   = np.mean(gen_rewards)
    gen_best_s   = gen_succs[gen_best_idx]
    gen_total_s  = sum(gen_succs)

    # SR eval
    gen_sr, gen_avg_r = eval_sr(gen_best_p, gen)

    flags = ""
    if gen_avg_r > best_rew_ever:
        best_rew_ever   = gen_avg_r
        best_params_rew = gen_best_p.copy()
        np.savez(SAVE_BEST_REW, best_params=best_params_rew,
                 reward=best_rew_ever, gen=gen)
        flags += " ★REW"

    if gen_sr > best_sr_ever:
        best_sr_ever   = gen_sr
        best_params_sr = gen_best_p.copy()
        np.savez(SAVE_BEST_SR, best_params=best_params_sr,
                 sr=best_sr_ever, gen=gen)
        np.savez(SAVE_AGENT, best_params=best_params_sr,
                 sr=best_sr_ever, gen=gen)
        flags += " ★SR"

    if (gen + 1) % SAVE_EVERY == 0 or gen == 0 or gen == N_GENERATIONS - 1:
        np.savez(SAVE_LATEST, best_params=gen_best_p,
                 gen=gen + 1, sigma=es.sigma)

    gen_time = time.time() - gen_t0
    eta_hr   = ((N_GENERATIONS - gen - 1) * gen_time) / 3600

    print(f"Gen {gen+1:3d}/{N_GENERATIONS} | {stage_str} | "
          f"Succ:{gen_total_s}/{POP_SIZE*EVALS_PER_IND} "
          f"BestS:{gen_best_s}/{EVALS_PER_IND} | "
          f"SR:{gen_sr:.0%}(best:{best_sr_ever:.0%}) | "
          f"AvgR:{gen_avg_r:8.0f} BestR*:{best_rew_ever:8.0f} | "
          f"σ:{es.sigma:.3f} | {gen_time/60:.1f}m | ETA:{eta_hr:.1f}h{flags}")

    log_data.append({
        'gen': gen+1, 'total_succ': gen_total_s, 'best_succ': gen_best_s,
        'sr': gen_sr, 'best_sr': best_sr_ever,
        'avg_r': gen_avg_r, 'best_rew': best_rew_ever,
        'sigma': es.sigma, 'gen_time_s': gen_time,
        'diffs': diffs_in_play,
    })
    if (gen + 1) % 5 == 0:
        with open(SAVE_LOG, 'w') as f:
            json.dump(log_data, f, indent=2)

    gen += 1

np.savez(SAVE_LATEST, best_params=best_params_sr, gen=gen, sigma=es.sigma)
np.savez(SAVE_AGENT, best_params=best_params_sr, sr=best_sr_ever, gen=gen)
with open(SAVE_LOG, 'w') as f:
    json.dump(log_data, f, indent=2)

elapsed = (time.time() - t0) / 3600
print(f"\n{'='*70}")
print(f"  Done in {elapsed:.1f}h | Best SR: {best_sr_ever:.0%}")
print(f"  Best reward: {best_rew_ever:.0f} | Weights → {SAVE_AGENT}")
print(f"{'='*70}")