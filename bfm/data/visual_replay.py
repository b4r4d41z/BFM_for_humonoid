"""Small arrays in RAM, RGB fetched only for sampled rows (deduplicated per batch)."""
import numpy as np
import torch
from bfm.contracts.vision import CAMERAS
from .replay_buffer import BFMReplayBuffer


class VisualReplayBuffer(BFMReplayBuffer):
    def __init__(self, dataset, seed=0, cameras=CAMERAS):
        super().__init__(dataset.episodes, seed)
        self.dataset = dataset
        self.cameras = tuple(cameras)
        self.feature_cache = None

    def observations(self, indices, prefix='obs'):
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError('Visual replay expects a flat batch of indices')
        indices = np.ascontiguousarray(indices)
        key = 'observation' if prefix == 'obs' else 'next_observation'
        result = {'proprio': self.data[key][indices].clone()}
        if self.feature_cache is not None:
            if self.feature_cache.cameras != self.cameras:
                raise ValueError('Feature cache camera ordering differs from replay')
            result.update(self.feature_cache.observations(indices, prefix))
            return result
        # Canonical cameras may differ in resolution. Keep a named tensor per view.
        for camera in self.cameras:
            unique, inverse = np.unique(indices, return_inverse=True)
            frames = []
            for i in unique:
                episode, row = self.dataset.image_rows[i]
                frames.append(self.dataset.roots[episode][f'{prefix}/images/{camera}'][row])
            shapes = {frame.shape for frame in frames}
            if len(shapes) != 1:
                raise ValueError('Batch mixes camera resolutions; use a consistent capture configuration')
            result[camera] = torch.from_numpy(np.stack(frames)[inverse])
        return result

    def sample_transitions(self, batch_size):
        indices = self.rng.integers(0, self.offsets[-1], size=batch_size)
        batch = {k: v[indices].clone() for k, v in self.data.items()
                 if k not in ('observation', 'next_observation', 'evaluation_state')}
        batch.update(observation=self.observations(indices), next_observation=self.observations(indices, 'next_obs'))
        return batch

    def reference_states(self, count, seed):
        indices = np.random.default_rng(seed).choice(self.offsets[-1], min(count, self.offsets[-1]), replace=False)
        # Small privileged sidecar is used only to construct task rewards.
        return self.data['evaluation_state'][indices].clone(), indices.tolist()

    def sample_states(self, batch_size):
        return self.observations(self.rng.integers(0, self.offsets[-1], batch_size), 'next_obs')

    sample_goal_states = sample_states

    def sample_sequences(self, batch_size, seq_len):
        raise NotImplementedError('Visual FB uses transition batches; temporal RGB loading is not implemented')
