import time
import numpy as np
import torch

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


# ---------- Evaluation ----------

def evaluate(env_cls, agent_p0, agent_p1, episodes=200):
    wins = np.zeros(3, dtype=int)  # [P0, P1, tie]
    for _ in range(episodes):
        env = env_cls()
        env.reset()
        obs = env.get_observation()
        mask = env.get_mask()
        done = env.terminal
        while not done:
            if env.current == 0:
                a = agent_p0.act(obs, mask, use_average=True)
            else:
                a = agent_p1.act(obs, mask, use_average=True)
            next_obs = env.step(a)
            done = env.terminal
            if not done:
                obs  = next_obs if next_obs is not None else env.get_observation()
                mask = env.get_mask()
        r0, r1 = env.get_rewards()
        if r0 > r1: wins[0] += 1
        elif r1 > r0: wins[1] += 1
        else: wins[2] += 1
    return wins


def evaluate_both_seats(env_cls, a0, a1, episodes=400):
    def one_side(p0, p1, n):
        w = np.zeros(3, dtype=int)
        for _ in range(n):
            env = env_cls()
            env.reset()
            obs = env.get_observation()
            mask = env.get_mask()
            done = env.terminal
            while not done:
                if env.current == 0:
                    act = p0.act(obs, mask, use_average=True)
                else:
                    act = p1.act(obs, mask, use_average=True)
                _ = env.step(act)
                done = env.terminal
                if not done:
                    obs = env.get_observation()
                    mask = env.get_mask()
            r0, r1 = env.get_rewards()
            if r0 > r1: w[0]+=1
            elif r1 > r0: w[1]+=1
            else: w[2]+=1
        return w

    half = episodes // 2
    wA = one_side(a0, a1, half)  # agent0 seat 0, agent1 seat 1
    wB = one_side(a1, a0, half)  # swapped
    return np.array([wA[0] + wB[1], wA[1] + wB[0], wA[2] + wB[2]])


# ---------- Training ----------

def main(seed=42, iters=5000, trajs_per_iter=64, batch_size=2048, device="cpu"):
    set_seed(seed)
    env = LeducEnv()
    obs_dim = env.get_observation().shape[0]
    act_dim = 4  # Call, Raise, Fold, Check

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

    single = evaluate(LeducEnv, agent0, agent1, episodes=1000)
    both   = evaluate_both_seats(LeducEnv, agent0, agent1, episodes=1000)

    sp0, sp1, st = pct_triple(single)
    ap0, ap1, at = pct_triple(both)

    print("Single-seat (P0 vs P1)")
    print(f"  P0/P1/T: {sp0:5.1f}% / {sp1:5.1f}% / {st:5.1f}%   (counts {fmt_counts(single)})")

    print("\nSeat-averaged (Agent0 vs Agent1)")
    print(f"  A0/A1/T: {ap0:5.1f}% / {ap1:5.1f}% / {at:5.1f}%   (counts {fmt_counts(both)})")
    print("="*80)


if __name__ == "__main__":
    main()
