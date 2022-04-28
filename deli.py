from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Union

import flax.core
import gym
import jax.numpy as jnp
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from buffer import TrajectoryBuffer
from misc import get_sa_dim
from save_utils import save_to_zip_file, load_from_zip_file, recursive_getattr, recursive_setattr

Params = flax.core.FrozenDict[str, Any]


class Deli(object):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        seed: int,
        dropout: float,
        tensorboard_log: Union[str, None],
        expert_goal: bool
    ):
        self.env = env

        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.observation_dim, self.action_dim = get_sa_dim(env)

        self.env_name = env.unwrapped.spec.id

        self.learning_rate = learning_rate
        self.seed = seed
        self.dropout = dropout
        self.replay_buffer: TrajectoryBuffer = None
        self.expert_buffer: TrajectoryBuffer = None
        self.normalize = 0.0
        self.tensorboard_log = tensorboard_log
        self.expert_goal = expert_goal

        self._writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.offline_rounds = 0
        self.n_updates = 0
        self.num_timesteps = 0
        self.diagnostics = defaultdict(list)

    def load_data(self, data_path: str, use_jax: bool = False):
        self.replay_buffer = TrajectoryBuffer(
            data_path=data_path,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            normalize=True,
            use_jax=use_jax
        )
        env_name = data_path.split("/")[-1].split("-")[0]
        expert_data_path = "/workspace/expertdata/dttrajectory/" + env_name + "-expert-v2"
        # expert_data_path = data_path.replace(env_name, "expert")
        self.normalize = self.replay_buffer.normalizing_factor
        if self.expert_goal:
            self.expert_buffer = TrajectoryBuffer(
                data_path=expert_data_path,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                normalize=True,
                use_jax=True
            )

    def _dump_logs(self):
        print("=" * 80)
        print("Env:\t", self.env_name)
        print("n_updates:\t", self.n_updates)
        for k, v in self.diagnostics.items():
            print(f"{k}: \t{jnp.mean(jnp.array(v))}")
            self._writer.add_scalar(k, np.mean(np.array(v)), self.n_updates,)

        print(f"Save to {self.tensorboard_log}")
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
        self.diagnostics = defaultdict(list)
        self.offline_rounds += 1
        for _ in range(total_timesteps):
            train_infos = self.train(batch_size=batch_size)
            for k, v in train_infos.items():
                if "loss" in k:
                    self.diagnostics["loss/"+k].append(v)

    @abstractmethod
    def get_save_params(self):
        pass

    def save(self, path: str):
        # Codes from stable baselines3
        data = self.__dict__.copy()
        params_to_save = self.get_save_params()
        save_to_zip_file(path, data=data, params=params_to_save)

    @abstractmethod
    def get_load_params(self):
        pass

    @classmethod
    def load(cls, path: str):
        # Codes from stable baselines3
        data, params, pytorch_variables = load_from_zip_file(path)
        env = None
        if "env" in data:
            env = data["env"]

        model = cls(env, _init_setup_model=False)
        model.__dict__.update(data)

        model.setup_model()
        model.set_parameters(params, exact_match=True)

        return model

    def set_parameters(
        self,
        load_path_or_dict: Dict,
        exact_match: bool = True,
    ) -> None:
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self.get_load_params())
        updated_objects = set()

        for name in params.keys():
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")
            jax_model = attr.load_dict(params[name])
            recursive_setattr(self, name, jax_model)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )