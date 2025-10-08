# dream_main.py
import time
import numpy as np
import torch
from contextlib import contextmanager

from algorithms.dream import DreamAgent, set_seed
from environment.leduc_env import LeducEnv


# ---------- Small helpers for printing ----------

def pct_triple(triple):
    total = max(1, int(np.sum(triple)))
    return tuple(100.0 * x / total for x in triple)

def fmt_counts(triple):
    return f"{int(triple[0])}/{int(triple[1])}/{int(triple[2])}"

def fmt_pct(triple):
    p0, p1, t = pct_triple(triple)
    return f"{p0:5.1f}%/{p1:5.1f}%/{t:5.1f}%"

def fmt_pair_counts(buf0, buf1):
    def k(n):
        return f"{n/1000:.0f}k" if n >= 1000 else str(n)
    return f"{k(buf0)}/{k(buf1)}"


# ---------- Turn off ε during evaluation ----------

@contextmanager
def disable_exploration(*agents):
    old_eps = [ag.eps for ag in agents]
    try:
        for ag in agents:
            ag.eps = 0.0
        yield
    finally:
        for ag, e in zip(agents, old_eps):
            ag.eps = e


# ---------- Evaluation (self-play) ----------

@torch.inference_mode()
def evaluate(env_cls, agent_p0, agent_p1, episodes=200):
    wins = np.zeros(3, dtype=int)  # [P0, P1, tie]
    with disable_exploration(agent_p0, agent_p1):
        for _ in range(episodes):
            env = env_cls()
            env.reset()
            done = env.terminal
            while not done:
                obs = env.get_observation()
                mask = env.get_mask()
                if env.current == 0:
                    a = agent_p0.act(obs, mask, use_average=True)
                else:
                    a = agent_p1.act(obs, mask, use_average=True)
                _ = env.step(a)
                done = env.terminal
            r0, r1 = env.get_rewards()
            if r0 > r1: wins[0] += 1
            elif r1 > r0: wins[1] += 1
            else: wins[2] += 1
    return wins


@torch.inference_mode()
def evaluate_both_seats(env_cls, a0, a1, episodes=400):
    def one_side(p0, p1, n):
        w = np.zeros(3, dtype=int)
        for _ in range(n):
            env = env_cls()
            env.reset()
            done = env.terminal
            while not done:
                obs = env.get_observation()
                mask = env.get_mask()
                act = p0.act(obs, mask, use_average=True) if env.current == 0 else p1.act(obs, mask, use_average=True)
                _ = env.step(act)
                done = env.terminal
            r0, r1 = env.get_rewards()
            if r0 > r1: w[0]+=1
            elif r1 > r0: w[1]+=1
            else: w[2]+=1
        return w

    with disable_exploration(a0, a1):
        half = episodes // 2
        wA = one_side(a0, a1, half)  # agent0 seat 0, agent1 seat 1
        wB = one_side(a1, a0, half)  # swapped
    return np.array([wA[0] + wB[1], wA[1] + wB[0], wA[2] + wB[2]])


# ---------- Fixed-opponent policies & evals ----------

def policy_random(obs, mask):
    legal = np.where(mask > 0)[0]
    return int(np.random.choice(legal))

def policy_always_call(obs, mask):
    legal = np.where(mask > 0)[0]
    return 0 if 0 in legal else int(np.random.choice(legal))

def policy_always_raise(obs, mask):
    legal = np.where(mask > 0)[0]
    return 1 if 1 in legal else int(np.random.choice(legal))

@torch.inference_mode()
def evaluate_vs_fixed(env_cls, agent, opponent_policy, episodes=300, agent_seat=0):
    scores = np.zeros(3, dtype=int)  # [agent_wins, opp_wins, ties]
    with disable_exploration(agent):
        for _ in range(episodes):
            env = env_cls()
            env.reset()
            done = env.terminal
            while not done:
                obs = env.get_observation()
                mask = env.get_mask()
                if env.current == agent_seat:
                    a = agent.act(obs, mask, use_average=True)
                else:
                    a = opponent_policy(obs, mask)
                _ = env.step(a)
                done = env.terminal

            r0, r1 = env.get_rewards()
            agent_r = r0 if agent_seat == 0 else r1
            opp_r   = r1 if agent_seat == 0 else r0
            if agent_r > opp_r: scores[0] += 1
            elif opp_r > agent_r: scores[1] += 1
            else: scores[2] += 1
    return scores

@torch.inference_mode()
def evaluate_vs_fixed_both_seats(env_cls, agent, opponent_policy, episodes=600):
    half = episodes // 2
    return evaluate_vs_fixed(env_cls, agent, opponent_policy, half, agent_seat=0) + \
           evaluate_vs_fixed(env_cls, agent, opponent_policy, half, agent_seat=1)


# ---------- Training ----------

def main(seed=42, iters=5000, trajs_per_iter=64, batch_size=4096, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(seed)

    env = LeducEnv()
    obs_dim = env.get_observation().shape[0]
    act_dim = 4  # 0:Call, 1:Raise, 2:Fold, 3:Check  (adjust if different)

    agent0 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)
    agent1 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)

    # Equalize init to reduce early drift
    agent1.adv_net.load_state_dict(agent0.adv_net.state_dict())
    agent1.baseline.load_state_dict(agent0.baseline.state_dict())

    # Persistent frozen opponents (refreshed once per iter)
    opp0 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)
    opp1 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)

    # Reused rollout envs (one per seat)
    roll_env0 = LeducEnv()
    roll_env1 = LeducEnv()

    log_every = max(1, iters // 50)  # ~50 logs over the run
    t0 = time.time()

    print(
        " iter  |   time   |  eps  |   loss0   loss1  |  buffers   |"
        "   single-seat P0/P1/T (cnt)   |    seat-avg A0/A1/T (cnt)"
    )
    print("-"*100)

    for it in range(1, iters + 1):
        # Decay exploration
        agent0.eps = max(0.02, 0.10 * (1.0 - it / 2000.0))
        agent1.eps = max(0.02, 0.10 * (1.0 - it / 2000.0))

        # Refresh frozen opponents ONCE per iteration (cheap)
        opp0.adv_net.load_state_dict(agent1.adv_net.state_dict(), strict=False)
        opp0.baseline.load_state_dict(agent1.baseline.state_dict(), strict=False)
        opp1.adv_net.load_state_dict(agent0.adv_net.state_dict(), strict=False)
        opp1.baseline.load_state_dict(agent0.baseline.state_dict(), strict=False)

        # -------- Data collection (reusing envs) --------
        for _ in range(trajs_per_iter):
            agent0.outcome_sampling_traj(roll_env0, player_i=0, opponent=opp0)
            agent1.outcome_sampling_traj(roll_env1, player_i=1, opponent=opp1)

        # -------- Train --------
        m0 = agent0.train_step(batch_size=batch_size)
        m1 = agent1.train_step(batch_size=batch_size)

        # -------- Periodic evaluation --------
        if it % log_every == 0:
            with torch.inference_mode():
                elapsed = time.time() - t0
                single = evaluate(LeducEnv, agent0, agent1, episodes=120)
                both   = evaluate_both_seats(LeducEnv, agent0, agent1, episodes=240)
            print(
                f"[{it:05d}] | {elapsed:7.1f}s | {agent0.eps:4.2f} | "
                f"{m0['loss']:7.3f}  {m1['loss']:7.3f} | "
                f"{fmt_pair_counts(len(agent0.buffer), len(agent1.buffer)):>9} | "
                f"{fmt_pct(single)} ({fmt_counts(single)})  |  "
                f"{fmt_pct(both)} ({fmt_counts(both)})"
            )

    # -------- Final summary --------
    total_time = time.time() - t0
    print("\n" + "="*80)
    print(f"Training completed in {total_time:.1f}s")
    print("="*80)

    with torch.inference_mode():
        single = evaluate(LeducEnv, agent0, agent1, episodes=1000)
        both   = evaluate_both_seats(LeducEnv, agent0, agent1, episodes=1000)

    sp0, sp1, st = pct_triple(single)
    ap0, ap1, at = pct_triple(both)

    print("Single-seat (P0 vs P1)")
    print(f"  P0/P1/T: {sp0:5.1f}% / {sp1:5.1f}% / {st:5.1f}%   (counts {fmt_counts(single)})")

    print("\nSeat-averaged (Agent0 vs Agent1)")
    print(f"  A0/A1/T: {ap0:5.1f}% / {ap1:5.1f}% / {at:5.1f}%   (counts {fmt_counts(both)})")

    # -------- Fixed-opponent evaluations (seat-averaged) --------
    print("\n" + "="*80)
    print("Evaluation vs fixed opponents (seat-averaged)")
    print("="*80)

    with torch.inference_mode():
        vs_random = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_random,      episodes=600)
        vs_call   = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_always_call, episodes=600)
        vs_raise  = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_always_raise,episodes=600)

    def show(name, triple):
        print(f"{name:14s}: {fmt_pct(triple)}  (counts {fmt_counts(triple)})")

    show("Random",      vs_random)
    show("Always_Call", vs_call)
    show("Always_Raise",vs_raise)
    print("="*80)


if __name__ == "__main__":
    main()
