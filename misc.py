from typing import Tuple, Any

import flax.core
import flax.linen as nn
import gym
import jax.numpy as jnp

Params = flax.core.FrozenDict[str, Any]


def get_sa_dim(env: gym.Env) -> Tuple:
    return env.observation_space.shape[0], env.action_space.shape[0]


class DeliFeaturesExtractor(nn.Module):
    _observation_dim: int
    _latent_dim: int

    @property
    def features_dim(self) -> int:
        return self._observation_dim + self._latent_dim

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return observations.reshape((observations.shape[0], -1))


class FlattenExtractor(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return x.reshape((x.shape[0], -1))