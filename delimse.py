from functools import partial
from typing import Any, Dict, List

import flax.core
import gym
import jax.numpy as jnp
import jax.random
import optax

from core import _update_mse_jit, _update_mse_jit_flow, _update_mse_jit_goal_loss
from deli import Deli
from misc import DeliFeaturesExtractor
from model import Model
from networks import VariationalAutoEncoder, SASPredictor, MSEActor

Params = flax.core.FrozenDict[str, Any]


class DeliMGMSE(Deli):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        seed: int = 0,
        dropout: float = 0.1,
        history_len: int = 20,
        latent_dim: int = 8,
        grad_flow: bool = False,
        tensorboard_log: str = None,
        data_path: str = None,
        expert_goal: bool = False,
        _init_setup_model: bool = True,
        **actor_kwargs
    ):
        super(DeliMGMSE, self).__init__(
            env=env,
            learning_rate=learning_rate,
            seed=seed,
            dropout=dropout,
            tensorboard_log=tensorboard_log,
            expert_goal=expert_goal
        )
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.grad_flow = grad_flow

        if expert_goal:
            self.update_ft = _update_mse_jit_goal_loss
        else:
            self.update_ft = _update_mse_jit_flow if self.grad_flow else _update_mse_jit

        self.actor: Model = None
        self.ae: Model = None
        self.sas_predictor: Model = None

        self.actor_kwargs = actor_kwargs

        self.key = jax.random.PRNGKey(seed)

        if _init_setup_model:
            self.load_data(data_path)
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
        init_ae_input = jnp.repeat(init_history, repeats=self.history_len, axis=0)
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
        self.key, key = jax.random.split(self.key, 2)
        replay_data = self.replay_buffer.o_history_sample(
            batch_size=batch_size,
            history_len=self.history_len,
            st_future_len=7,
        )
        if self.expert_buffer is not None:
            goal_data = self.expert_buffer.sample(key=key, batch_size=batch_size)
            goal_observations = goal_data.observations[:, jnp.newaxis, :]
            goal_actions = goal_data.actions[:, jnp.newaxis, :]
        else:
            goal_observations = replay_data.st_future.observations
            goal_actions = replay_data.st_future.actions

        self.key, new_sas_predictor, new_ae, new_actor, infos = self.update_ft(
            rng=key,
            actor=self.actor,
            ae=self.ae,
            sas_predictor=self.sas_predictor,
            history_observations=replay_data.history.observations,
            history_actions=replay_data.history.actions,
            observations=replay_data.observations,
            actions=replay_data.actions,
            next_observations=replay_data.st_future.observations[:, 0, :][:, jnp.newaxis, :],
            goal_observations=goal_observations,
            goal_actions=goal_actions
        )
        self.sas_predictor = new_sas_predictor
        self.ae = new_ae
        self.actor = new_actor
        self.n_updates += 1

        return infos

    def get_save_params(self) -> Dict:
        params_dict = {
            "actor": self.actor.params,
            "ae": self.ae.params,
            "sas_predictor": self.sas_predictor.params
        }
        return params_dict

    def get_load_params(self) -> List:
        return ["actor", "ae", "sas_predictor"]
