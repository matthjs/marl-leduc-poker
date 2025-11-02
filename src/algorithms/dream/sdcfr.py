# algorithms/dream/sdcfr.py
import numpy as np
import torch

class SDCFROpponent:
    """
    Stores regret-net snapshots and, per episode, samples one snapshot
    with probability proportional to its iteration weight (Linear-CFR: w_t = t).
    Policy = regret matching on that snapshot's predicted advantages.
    """
    def __init__(self, net_ctor, device=None, eta=2e-3):
        """
        net_ctor: zero-arg callable returning a freshly constructed RegretNet
                  (must match your agent's RegretNet architecture).
        """
        self._ctor = net_ctor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._snap_nets = []   # list[nn.Module]
        self._weights = []     # list[float], Linear-CFR weights w_t = t
        self._eta = float(eta) # tiny prior
        self._ep_net = None    # selected snapshot for current episode

    def add_snapshot(self, iter_t: int, state_dict):
        net = self._ctor().to(self.device).eval()
        net.load_state_dict(state_dict)
        self._snap_nets.append(net)
        self._weights.append(float(iter_t))

    def _sample_episode_net(self):
        w = np.asarray(self._weights, dtype=np.float64)
        p = w / max(1e-12, w.sum())
        idx = int(np.random.choice(len(self._snap_nets), p=p))
        self._ep_net = self._snap_nets[idx]

    def reset_episode(self):
        self._ep_net = None

    @torch.inference_mode()
    def _rm_policy_with(self, regret_net, obs, mask):
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        adv   = regret_net(obs_t).squeeze(0).cpu().numpy()
        adv   = adv * mask
        pos   = np.maximum(adv, 0.0)
        prior = (mask > 0).astype(np.float32)
        dist  = pos + self._eta * prior
        dist *= prior
        s = float(dist.sum())
        if s <= 0:
            legal = np.where(mask > 0)[0]
            out = np.zeros_like(dist, dtype=np.float32)
            if len(legal) > 0:
                out[legal] = 1.0 / len(legal)
            return out
        return (dist / s).astype(np.float32)

    def policy(self, obs, mask):
        if self._ep_net is None:
            self._sample_episode_net()
        return self._rm_policy_with(self._ep_net, obs, mask)

    def act(self, obs, mask):
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0
        pi = self.policy(obs, mask)
        return int(np.random.choice(legal, p=pi[legal]))


@torch.inference_mode()
def evaluate_sdcfr(env_cls, s0: SDCFROpponent, s1: SDCFROpponent, episodes=200):
    wins = np.zeros(3, dtype=int)
    for _ in range(episodes):
        s0.reset_episode(); s1.reset_episode()
        env = env_cls(); env.reset()
        done = env.terminal
        while not done:
            obs = env.get_observation(); mask = env.get_mask()
            a = s0.act(obs, mask) if env.current == 0 else s1.act(obs, mask)
            env.step(a); done = env.terminal
        r0, r1 = env.get_rewards()
        if r0 > r1: wins[0] += 1
        elif r1 > r0: wins[1] += 1
        else: wins[2] += 1
    return wins

@torch.inference_mode()
def evaluate_sdcfr_both_seats(env_cls, s0: SDCFROpponent, s1: SDCFROpponent, episodes=400):
    def one_side(p0, p1, n):
        w = np.zeros(3, dtype=int)
        for _ in range(n):
            p0.reset_episode(); p1.reset_episode()
            env = env_cls(); env.reset()
            done = env.terminal
            while not done:
                obs = env.get_observation(); mask = env.get_mask()
                a = p0.act(obs, mask) if env.current == 0 else p1.act(obs, mask)
                env.step(a); done = env.terminal
            r0, r1 = env.get_rewards()
            if r0 > r1: w[0]+=1
            elif r1 > r0: w[1]+=1
            else: w[2]+=1
        return w
    half = episodes // 2
    wA = one_side(s0, s1, half)
    wB = one_side(s1, s0, half)
    return np.array([wA[0] + wB[1], wA[1] + wB[0], wA[2] + wB[2]])
