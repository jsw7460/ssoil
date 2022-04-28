from functools import partial
from typing import Any

import flax.core
import gym
import jax.numpy as jnp
import jax.random
import optax

from core import update_bc
from deli import Deli
from misc import DeliFeaturesExtractor
from model import Model
from networks import MSEActor

Params = flax.core.FrozenDict[str, Any]


class DeliBC(Deli):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        seed: int = 0,
        dropout: float = 0.0,
        tensorboard_log: str = None,
        augmentation: bool = True,
        data_path: str = None,
        **actor_kwargs
    ):
        super(DeliBC, self).__init__(
            env=env,
            learning_rate=learning_rate,
            seed=seed,
            dropout=dropout,
            tensorboard_log=tensorboard_log
        )
        self.actor: Model = None
        self.actor_kwargs = actor_kwargs

        self.key = jax.random.PRNGKey(seed)

        self.augmentation = augmentation
        self.setup_model()
        self.load_data(data_path, use_jax=True)

        self.data_sampling = self.replay_buffer.sample if not self.augmentation else self.replay_buffer.noise_sample

    def setup_model(self):
        net_arch = self.actor_kwargs.get("net_arch", None)

        init_observation = self.observation_space.sample()[jnp.newaxis, ...]

        dropout_key, actor_key = jax.random.split(self.key)

        # Actor
        feature_extractor = DeliFeaturesExtractor(self.observation_dim, 0)
        actor_def = MSEActor(
            features_extractor=feature_extractor,
            action_dim=self.action_dim,
            net_arch=net_arch,
            dropout=self.dropout,
        )

        init_latent = jax.random.normal(dropout_key, shape=init_observation.shape,)
        init_actor = jnp.concatenate((init_observation, init_latent), axis=1)

        init_actor_rngs = {"params": actor_key, "dropout": dropout_key}
        self.actor = Model.create(
            model_def=actor_def,
            inputs=[init_actor_rngs, init_actor],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _predict(self, observations: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        rng, dropout_key = jax.random.split(self.key)
        action = self.actor.apply_fn(
            {"params": self.actor.params},
            observations,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )
        return action

    def train(self, batch_size: int = 256,):
        self.key, buffer_key, update_key = jax.random.split(self.key, 3)

        replay_data = self.data_sampling(
            key=buffer_key,
            batch_size=batch_size,
        )

        new_actor, infos = update_bc(
            key=update_key,
            actor=self.actor,
            replay_data=replay_data,
        )
        self.actor = new_actor
        self.n_updates += 1
        return infos
