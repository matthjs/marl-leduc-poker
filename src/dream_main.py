import time
import numpy as np
import torch
from contextlib import contextmanager

from algorithms.dream import DreamAgent, set_seed
from environment.leduc_env import LeducEnv
from copy import deepcopy


# ---------- Opponent wrapper ----------
class AvgPolicyOpponent:
    def __init__(self, src_agent):
        self.avg_net = deepcopy(src_agent.avg_net).eval()
        self.device = next(self.avg_net.parameters()).device
        self.eps = 0.0

    @torch.inference_mode()
    def policy(self, obs, mask, use_average=True):
        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
        pi = self.avg_net(obs_t, mask_t).squeeze(0).cpu().numpy()
        return pi

    def act(self, obs, mask, use_average=True):
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0
        pi = self.policy(obs, mask, use_average=True)
        return int(np.random.choice(legal, p=pi[legal]))


# ---------- Formatting helpers ----------
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


# ---------- Disable exploration ----------
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


# ---------- Evaluation ----------
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
        wA = one_side(a0, a1, half)
        wB = one_side(a1, a0, half)
    return np.array([wA[0] + wB[1], wA[1] + wB[0], wA[2] + wB[2]])


# ---------- Fixed opponents ----------
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
    scores = np.zeros(3, dtype=int)
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
def main(
    seed=42,
    iters=1000,
    trajs_per_iter=64,
    batch_size=4096,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(seed)

    env = LeducEnv()
    obs_dim = env.get_observation().shape[0]
    act_dim = 4

    # Agents
    agent0 = DreamAgent(
        obs_dim, act_dim,
        lr=5e-4, lr_q=5e-4, device=device,
    )
    agent1 = DreamAgent(
        obs_dim, act_dim,
        lr=5e-4, lr_q=5e-4, device=device,
    )

    # Sync initialization
    agent1.adv_net.load_state_dict(agent0.adv_net.state_dict())
    agent1.q_net.load_state_dict(agent0.q_net.state_dict())

    opp0 = AvgPolicyOpponent(agent1)
    opp1 = AvgPolicyOpponent(agent0)

    roll_env0, roll_env1 = LeducEnv(), LeducEnv()
    log_every = max(1, iters // 50)
    t0 = time.time()

    print(" iter  |   time   |  eps  |   loss0   loss1  | buffers   |"
          "   single-seat P0/P1/T (cnt)   |    seat-avg A0/A1/T (cnt)")
    print("-"*100)

    for it in range(1, iters + 1):
        # ε-decay
        agent0.eps = max(0.005, 0.10 * (1.0 - it / 2000.0))
        agent1.eps = max(0.005, 0.10 * (1.0 - it / 2000.0))

        # Refresh frozen opponents
        if it % 50 == 0:
            opp0 = AvgPolicyOpponent(agent1)
            opp1 = AvgPolicyOpponent(agent0)

        # Data collection
        for _ in range(trajs_per_iter):
            old_eps0, old_eps1 = agent0.eps, agent1.eps
            agent0.eps = max(0.25, agent0.eps)
            agent1.eps = max(0.25, agent1.eps)

            if np.random.rand() < 0.5:
                agent0.outcome_sampling_traj(roll_env0, player_i=0, opponent=opp0)
                agent1.outcome_sampling_traj(roll_env1, player_i=1, opponent=opp1)
            else:
                agent0.outcome_sampling_traj(roll_env0, player_i=1, opponent=opp0)
                agent1.outcome_sampling_traj(roll_env1, player_i=0, opponent=opp1)

            agent0.eps, agent1.eps = old_eps0, old_eps1

        # Train
        m0 = agent0.train_step(batch_size=batch_size)
        m1 = agent1.train_step(batch_size=batch_size)

        # CFR iteration increment
        agent0.increment_iteration()
        agent1.increment_iteration()

        # Evaluation
        if it % log_every == 0:
            with torch.inference_mode():
                elapsed = time.time() - t0
                single = evaluate(LeducEnv, agent0, agent1, episodes=120)
                both = evaluate_both_seats(LeducEnv, agent0, agent1, episodes=240)
            print(
                f"[{it:05d}] | {elapsed:7.1f}s | {agent0.eps:4.2f} | "
                f"{m0['loss']:7.3f}  {m1['loss']:7.3f} | "
                f"{fmt_pair_counts(len(agent0.buffer), len(agent1.buffer)):>9} | "
                f"{fmt_pct(single)} ({fmt_counts(single)})  |  "
                f"{fmt_pct(both)} ({fmt_counts(both)})"
            )

    total_time = time.time() - t0
    print("\n" + "="*80)
    print(f"Training completed in {total_time:.1f}s")
    print("="*80)

    # Final evaluation
    with torch.inference_mode():
        single = evaluate(LeducEnv, agent0, agent1, episodes=200)
        both = evaluate_both_seats(LeducEnv, agent0, agent1, episodes=400)
    print(f"\nFinal single-seat results: {fmt_pct(single)} ({fmt_counts(single)})")
    print(f"Final seat-averaged results: {fmt_pct(both)} ({fmt_counts(both)})")

    # Evaluate exploitability vs fixed opponents
    print("\nEvaluating exploitability vs fixed opponents...")
    fixed_random = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_random)
    fixed_call = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_always_call)
    print(f"Against random:        {fmt_pct(fixed_random)} ({fmt_counts(fixed_random)})")
    print(f"Against always-call:   {fmt_pct(fixed_call)} ({fmt_counts(fixed_call)})")
    fixed_raise = evaluate_vs_fixed_both_seats(LeducEnv, agent0, policy_always_raise)
    print(f"Against always-raise:  {fmt_pct(fixed_raise)} ({fmt_counts(fixed_raise)})")


if __name__ == "__main__":
    main()
