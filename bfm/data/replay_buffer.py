import copy
import numpy as np
import torch


class BFMReplayBuffer:
    def __init__(self, episodes, seed=0):
        if not episodes:
            raise ValueError("Replay needs at least one episode")
        for e in episodes:
            n = len(e["observation"])
            term, trunc = torch.as_tensor(e["terminated"]), torch.as_tensor(e["truncated"])
            if n == 0 or any(len(v) != n for v in e.values()):
                raise ValueError("Replay episode arrays must have matching nonzero lengths")
            if term.dtype != torch.bool or trunc.dtype != torch.bool or (term & trunc).any():
                raise ValueError("Invalid replay terminal flags")
            ends = term | trunc
            if ends[:-1].any() or not ends[-1]:
                raise ValueError("Replay expects complete individual episodes")
            if not torch.allclose(torch.as_tensor(e["next_observation"][:-1]), torch.as_tensor(e["observation"][1:]), atol=1e-6, rtol=1e-6):
                raise ValueError("Replay next observations cross a reset boundary")
        self.lengths = [len(e["observation"]) for e in episodes]
        self.offsets = np.cumsum([0] + self.lengths)
        self.data = {k: torch.cat([torch.as_tensor(e[k]) for e in episodes]) for k in episodes[0]}
        self.rng = np.random.default_rng(seed)

    def sample_transitions(self, batch_size):
        idx = self.rng.integers(0, self.offsets[-1], size=batch_size)
        return {k: v[idx].clone() for k, v in self.data.items()}

    def sample_states(self, batch_size):
        # Same reference distribution rho as the FB in-batch next-state columns.
        idx = self.rng.integers(0, self.offsets[-1], size=batch_size)
        return self.data["next_observation"][idx].clone()

    sample_goal_states = sample_states

    def reference_states(self, count, seed):
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.offsets[-1], min(count, self.offsets[-1]), replace=False)
        return self.data["next_observation"][idx].clone(), idx.tolist()

    def sample_sequences(self, batch_size, seq_len):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        starts = np.concatenate([np.arange(o, o + max(0, n - seq_len + 1)) for o, n in zip(self.offsets[:-1], self.lengths)])
        if not len(starts):
            raise ValueError("No episode long enough for requested sequence")
        idx = self.rng.choice(starts, batch_size)[:, None] + np.arange(seq_len)[None, :]
        return {k: v[idx].clone() for k, v in self.data.items()}

    def state_dict(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state)
