from typing import Union

import gym
import jax
import jax.numpy as jnp

from delimg import DeliMGMSE, DeliMGMLE
from model import Model
from functools import partial


class DeliSampler(object):
    def __init__(
        self,
        seed: int,
        latent_dim: int,
        vae: Model,
        normalizing_factor: float,
        history_len: int = 30,
    ):
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        self.history_observation = []
        self.history_action = []

        self.latent_dim = latent_dim
        self.vae: Model = vae
        self.normalizing_factor = normalizing_factor
        self.history_len = history_len

        self.observation_dim = None
        self.action_dim = None

    def __len__(self):
        return len(self.history_observation)

    def normalize_obs(self, observation: jnp.ndarray):
        return observation.copy() / self.normalizing_factor

    @partial(jax.jit, static_argnums=(0, ))
    def get_history_latent(self):
        if len(self) == 0:
            self.key, normal_key = jax.random.split(self.key, 2)
            latent = jax.random.normal(normal_key, shape=(self.latent_dim, ))

        else:
            self.key, dropout_key, decoder_key = jax.random.split(self.key, 3)
            history_obs = jnp.vstack(self.history_observation)[-self.history_len:, ...]
            history_act = jnp.vstack(self.history_action)[-self.history_len:, ...]

            cur_hist_len = len(history_obs)
            hist_padding_obs = jnp.zeros((self.history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = jnp.zeros((self.history_len - cur_hist_len, self.action_dim))

            history_obs = jnp.vstack((hist_padding_obs, history_obs))
            history_act = jnp.vstack((hist_padding_act, history_act))

            vae_input = jnp.hstack((history_obs, history_act))
            _, latent, *_ = self.vae.apply_fn(
                {"params": self.vae.params},
                vae_input,
                deterministic=True,
                rngs={"dropout": dropout_key, "decoder": decoder_key}
            )

        return jnp.squeeze(latent)

    def append(self, observation: jnp.ndarray, action: jnp.ndarray) -> None:
        self.observation_dim = observation.shape[-1]
        self.action_dim = action.shape[-1]

        if observation.ndim == 1:
            observation = observation[jnp.newaxis, ...]
        self.history_observation.append(observation.copy())
        self.history_action.append(action.copy())

    def reset(self):
        self.history_observation = []
        self.history_action = []

    def get_delig_policy_input(self, observation: jnp.ndarray):
        history_latent = self.get_history_latent()
        policy_input = jnp.hstack((observation, history_latent))[jnp.newaxis, ...]
        return policy_input


def evaluate_deli(
    seed: int,
    env: gym.Env,
    model: Union[DeliMGMSE, DeliMGMLE],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
):
    history_len = model.history_len
    normalizing_factor = model.replay_buffer.normalizing_factor
    latent_dim = model.latent_dim
    vae = model.ae

    sampler = DeliSampler(seed, latent_dim, vae, normalizing_factor, history_len)
    episodic_rewards = []
    episodic_lengths = []

    for i in range(n_eval_episodes):
        sampler.reset()
        observation = env.reset()
        observation = sampler.normalize_obs(observation)
        dones = False
        current_rewards = 0
        current_lengths = 0
        while not dones:
            current_lengths += 1
            policy_input = sampler.get_delig_policy_input(observation)
            action = model._predict(policy_input, deterministic=deterministic)
            if action.ndim == 0:
                action = action[jnp.newaxis, ...]

            next_observation, rewards, dones, infos = env.step(action)
            sampler.append(observation, action)
            current_rewards += rewards

            observation = sampler.normalize_obs(next_observation)

        episodic_rewards.append(current_rewards)
        episodic_lengths.append(current_lengths)

    return jnp.mean(jnp.array(episodic_rewards)), jnp.mean(jnp.array(episodic_lengths))

