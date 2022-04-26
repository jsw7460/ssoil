from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Tuple, Any

import flax.core
import flax.linen as nn
import gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter

from buffer import TrajectoryBuffer
from core import _update_mse_jit, _update_mse_jit_flow, _update_mle_jit
from model import Model
from networks import VariationalAutoEncoder, WassersteinAutoEncoder, SASPredictor, MSEActor, MLEActor

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


class Deli(object):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        seed: int = 0,
        dropout: float = 0.1,
        tensorboard_log: str = None,
    ):
        self.env = env

        self.env_name = env.unwrapped.spec.id

        self.learning_rate = learning_rate
        self.seed = seed
        self.dropout = dropout
        self.replay_buffer: TrajectoryBuffer = None
        self.tensorboard_log = tensorboard_log

        self._writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.offline_rounds = 0
        self.n_updates = 0
        self.num_timesteps = 0
        self.diagnostics = defaultdict(list)

    def load_data(self, data_path: str):
        self.replay_buffer = TrajectoryBuffer(
            data_path=data_path,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            normalize=True,
        )

    def _dump_logs(self):
        print("=" * 80)
        print("Env:\t", self.env_name)
        print("n_updates:\t", self.n_updates)
        for k, v in self.diagnostics.items():
            print(f"{k}: \t{jnp.mean(jnp.array(v))}")
            self._writer.add_scalar(k, np.mean(np.array(v)), self.n_updates)
        self._writer.flush()

        self._writer.close()

        # Init
        self.diagnostics = defaultdict(list)

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def _predict(self, observations: jnp.ndarray, deterministic: bool):
        pass

    @abstractmethod
    def train(self, batch_size: int):
        pass

    def learn(self, total_timesteps: int, batch_size: int = 256):
        self.offline_rounds += 1
        for _ in range(total_timesteps):
            train_infos = self.train(batch_size=batch_size)
            for k, v in train_infos.items():
                if "loss" in k:
                    self.diagnostics[k].append(v)


class DeliMGMSE(Deli):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        seed: int = 0,
        dropout: float = 0.1,
        history_len: int = 20,
        latent_dim: int = 25,
        grad_flow: bool = False,
        tensorboard_log: str = None,
        **actor_kwargs
    ):
        super(DeliMGMSE, self).__init__(
            env=env,
            learning_rate=learning_rate,
            seed=seed,
            dropout=dropout,
            tensorboard_log=tensorboard_log
        )
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.grad_flow = grad_flow

        self.update_ft = _update_mse_jit_flow if self.grad_flow else _update_mse_jit

        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.observation_dim, self.action_dim = get_sa_dim(env)

        self.actor: Model = None
        self.ae: Model = None
        self.sas_predictor: Model = None

        self.actor_kwargs = actor_kwargs

        self.key = jax.random.PRNGKey(seed)

        self.setup_model()

    def setup_model(self):
        net_arch = self.actor_kwargs.get("net_arch", None)

        init_latent = jax.random.normal(self.key, shape=(1, self.latent_dim))
        init_observation = self.observation_space.sample()[jnp.newaxis, ...]
        init_action = self.action_space.sample()[jnp.newaxis, ...]
        init_history = jnp.concatenate((init_observation, init_action), axis=1)

        dropout_key, actor_key, ae_key, sas_key = jax.random.split(self.key, num=4)

        # Actor
        feature_extractor = DeliFeaturesExtractor(self.observation_dim, self.latent_dim)
        actor_def = MSEActor(
            features_extractor=feature_extractor,
            action_dim=self.action_dim,
            net_arch=net_arch,
            dropout=self.dropout,
        )

        init_actor_input = jnp.concatenate((init_observation, init_latent), axis=1)
        init_actor_rngs = {"params": actor_key, "dropout": dropout_key}
        self.actor = Model.create(
            model_def=actor_def,
            inputs=[init_actor_rngs, init_actor_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # AE
        # ae_def = WassersteinAutoEncoder(            # Or Variational Autoencoder
        #     state_dim=self.observation_dim,
        #     latent_dim=self.latent_dim,
        #     squashed_output=True,
        # )
        ae_def = VariationalAutoEncoder(
            state_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            squashed_output=True
        )
        init_ae_input = jnp.repeat(init_history, repeats=self.history_len, axis=0, )
        ae_key, dropout_key, decoder_key, noise_key = jax.random.split(ae_key, 4)
        init_ae_rngs = {"params": ae_key, "dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
        self.ae = Model.create(
            model_def=ae_def,
            inputs=[init_ae_rngs, init_ae_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # SAS predictor
        sas_def = SASPredictor(
            state_dim=self.observation_dim,
            net_arch=net_arch,
            dropout=self.dropout
        )
        init_action = self.action_space.sample()[jnp.newaxis, ...]
        init_sas_input = jnp.hstack((init_observation, init_action))
        init_sas_rng = {"params": sas_key, "dropout": dropout_key}

        self.sas_predictor = Model.create(
            model_def=sas_def,
            inputs=[init_sas_rng, init_sas_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _predict(self, observations: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 여기서의 observation은 latent가 포함되어있는 말그대로 input 자체

        rng, dropout_key = jax.random.split(self.key)
        action = self.actor.apply_fn(
            {"params": self.actor.params},
            observations,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )
        return action

    def train(self, batch_size: int = 256,):
        replay_data = self.replay_buffer.history_sample(
            batch_size=batch_size,
            history_len=self.history_len,
            st_future_len=7
        )
        self.key, key = jax.random.split(self.key, 2)

        self.key, new_sas_predictor, new_ae, new_actor, infos = self.update_ft(
            rng=key,
            actor=self.actor,
            ae=self.ae,
            sas_predictor=self.sas_predictor,
            replay_data=replay_data,
        )
        self.sas_predictor = new_sas_predictor
        self.ae = new_ae
        self.actor = new_actor
        self.n_updates += 1

        return infos


class DeliMGMLE(Deli):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        seed: int = 0,
        dropout: float = 0.1,
        history_len: int = 20,
        latent_dim: int = 25,
        **actor_kwargs
    ):
        super(DeliMGMLE, self).__init__(
            env=env,
            learning_rate=learning_rate,
            seed=seed,
            dropout=dropout,
        )
        self.history_len = history_len
        self.latent_dim = latent_dim

        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.observation_dim, self.action_dim = get_sa_dim(env)

        self.actor: Model = None
        self.vae: Model = None
        self.sas_predictor: Model = None

        self.actor_kwargs = actor_kwargs

        self.key = jax.random.PRNGKey(seed)

        self.setup_model()

    def setup_model(self):
        net_arch = self.actor_kwargs.get("net_arch", None)

        init_latent = jax.random.normal(self.key, shape=(1, self.latent_dim))
        init_observation = self.observation_space.sample()[jnp.newaxis, ...]
        init_action = self.action_space.sample()[jnp.newaxis, ...]
        init_history = jnp.concatenate((init_observation, init_action), axis=1)

        actor_key, vae_key, sas_key = jax.random.split(self.key, num=3)

        # Actor
        feature_extractor = DeliFeaturesExtractor(self.observation_dim, self.latent_dim)
        actor_def = MLEActor(
            features_extractor=feature_extractor,
            action_dim=self.action_dim,
            dropout=self.dropout,
        )

        actor_key, dropout_key, action_sample = jax.random.split(actor_key, 3)
        init_actor_input = jnp.concatenate((init_observation, init_latent), axis=1)

        init_actor_rngs = {"params": actor_key, "dropout": dropout_key, "action_sample": action_sample}
        self.actor = Model.create(
            model_def=actor_def,
            inputs=[init_actor_rngs, init_actor_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # VAE
        vae_def = VariationalAutoEncoder(
            state_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            squashed_output=True,
        )
        init_vae_input = jnp.repeat(init_history, repeats=self.history_len, axis=0, )
        vae_key, dropout_key, decoder_key = jax.random.split(vae_key, 3)
        init_vae_rngs = {"params": vae_key, "dropout": dropout_key, "decoder": decoder_key}
        self.vae = Model.create(
            model_def=vae_def,
            inputs=[init_vae_rngs, init_vae_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # SAS predictor
        sas_def = SASPredictor(
            state_dim=self.observation_dim,
            net_arch=net_arch,
            dropout=self.dropout
        )
        init_action = self.action_space.sample()[jnp.newaxis, ...]
        init_sas_input = jnp.hstack((init_observation, init_action))
        init_sas_rng = {"params": sas_key, "dropout": dropout_key}
        self.sas_predictor = Model.create(
            model_def=sas_def,
            inputs=[init_sas_rng, init_sas_input],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

    def _predict(self, observations: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # 여기서의 observation은 latent가 포함되어있는 말그대로 input 자체

        self.key, dropout_key, action_sample_key = jax.random.split(self.key, 3)
        action = self.actor.apply_fn(
            {"params": self.actor.params},
            observations,
            deterministic=deterministic,
            rngs={"dropout": dropout_key, "action_sample": action_sample_key}
        )
        return action

    def train(self, batch_size: int = 256):
        replay_data = self.replay_buffer.history_sample(
            batch_size=batch_size,
            history_len=self.history_len,
            st_future_len=7
        )
        self.key, key = jax.random.split(self.key, 2)

        new_key, new_sas_predictor, new_vae, new_actor, infos = _update_mle_jit(
            rng=key,
            actor=self.actor,
            vae=self.vae,
            sas_predictor=self.sas_predictor,
            replay_data=replay_data,
        )
        self.key = new_key
        self.sas_predictor = new_sas_predictor
        self.vae = new_vae
        self.actor = new_actor

        self.n_updates += 1

        return infos
