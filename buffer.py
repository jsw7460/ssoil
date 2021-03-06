from abc import ABC, abstractmethod
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

np.random.seed(0)


class History(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray


Future = History
STFuture = Future
LTFuture = Future


class STermSubtrajBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    history: History
    st_future: STFuture


class StateActionBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray


class ReplayBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class TrajectoryBuffer(BaseBuffer):
    def __init__(
        self,
        data_path: str,
        observation_dim: int,
        action_dim: int,
        normalize: bool = True,
        limit: int = -1,
        use_jax: bool = False,
    ):
        import pickle
        with open(data_path + ".pkl", "rb") as f:
            expert_dataset = pickle.load(f)

        buffer_size = len(expert_dataset)
        max_traj_len = max([len(traj["observations"]) for traj in expert_dataset])
        self.use_terminal = "terminals" in expert_dataset[0]

        super(TrajectoryBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_dim=observation_dim,
            action_dim=action_dim
        )

        self.use_jax = use_jax

        self.expert_dataset = expert_dataset
        self.max_traj_len = max_traj_len
        self.normalize = normalize
        self.normalizing_factor = None
        self.limit = limit

        self.observation_traj = None
        self.action_traj = None
        self.reward_traj = None
        self.terminal_traj = None
        self.traj_lengths = None

        self.reset()

    def reset(self):
        self.observation_traj = np.zeros((self.buffer_size, self.max_traj_len, self.observation_dim))
        self.action_traj = np.zeros((self.buffer_size, self.max_traj_len, self.action_dim))
        self.reward_traj = np.zeros((self.buffer_size, self.max_traj_len))
        if self.use_terminal:
            self.terminal_traj = np.zeros((self.buffer_size, self.max_traj_len))
        self.traj_lengths = np.zeros((self.buffer_size, 1))

        for traj_idx in range(self.buffer_size):
            traj_data = self.expert_dataset[traj_idx]
            cur_traj_len = len(traj_data["rewards"])

            self.observation_traj[traj_idx, :cur_traj_len, :] = traj_data["observations"].copy()
            self.action_traj[traj_idx, :cur_traj_len, :] = traj_data["actions"].copy()
            self.reward_traj[traj_idx, :cur_traj_len] = traj_data["rewards"].copy()
            if self.use_terminal:
                self.terminal_traj[traj_idx, :cur_traj_len] = traj_data["terminals"].copy()
            self.traj_lengths[traj_idx, ...] = cur_traj_len

        if self.normalize:
            max_obs = np.max(self.observation_traj)
            min_obs = np.min(self.observation_traj)
            self.normalizing_factor = np.max([max_obs, -min_obs])
        else:
            self.normalizing_factor = 1.0

        if self.limit > 0:
            assert self.use_jax, "Landmark?????? ??? ?????? ?????? ????????????"
            self.observation_traj = self.observation_traj[:self.limit, ...]
            self.action_traj = self.action_traj[:self.limit, ...]

        if self.use_jax:
            self.observation_traj = jnp.array(self.observation_traj)
            self.action_traj = jnp.array(self.action_traj)
        self.observation_traj /= self.normalizing_factor

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_only_final_state(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        # With Gaussian noise
        batch_key, timestep_key, normal_key = jax.random.split(key, 3)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=(self.traj_lengths - 1).squeeze(),
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        # With Gaussian noise
        batch_key, timestep_key, normal_key = jax.random.split(key, 3)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size, ), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size, ),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        # noise = jax.random.normal(normal_key, shape=current_observations.shape) * 3e-4
        # current_observations = current_observations + noise

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2))
    def noise_sample(self, key: jnp.ndarray, batch_size: int):
        # Return just state-action samples
        batch_key, timestep_key = jax.random.split(key, 2)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        # batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        # timesteps = np.random.randint(0, self.traj_lengths - 1)[batch_inds].squeeze()

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        return StateActionBufferSample(current_observations, current_actions)

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def _history_sample(self, key: jnp.ndarray, batch_size: int):
        batch_key, timestep_key = jax.random.split(key)
        batch_inds = jax.random.randint(batch_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        timesteps = jax.random.randint(
            timestep_key,
            shape=(self.buffer_size,),
            minval=0,
            maxval=(self.traj_lengths - 1).squeeze()
        )
        timesteps = timesteps[batch_inds]

        return batch_inds, timesteps

    def history_sample(self, key, batch_size: int, history_len: int, st_future_len: int = None, k_nn: int = 0):
        batch_inds, timesteps = self._history_sample(key, batch_size)

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        history_observations, history_actions = [], []
        st_future_observations, st_future_actions = [], []

        for timestep, batch in zip(timesteps, batch_inds):
            start_pts = np.max([timestep - history_len, 0])
            hist_obs = self.observation_traj[batch, start_pts: timestep, ...]
            hist_act = self.action_traj[batch, start_pts: timestep, ...]
            cur_hist_len = len(hist_obs)        # ???????????? timestep??? ????????????, ????????? ?????? history_len?????? ????????????.

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            history_observations.append(hist_obs)
            history_actions.append(hist_act)

            st_fut_obs = self.observation_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            st_future_observations.append(st_fut_obs)
            st_future_actions.append(st_fut_act)

        history_observations = jnp.vstack(history_observations).reshape(batch_size, history_len, -1)
        history_actions = jnp.vstack(history_actions).reshape(batch_size, history_len, -1)

        st_future_observations = jnp.vstack(st_future_observations).reshape(batch_size, st_future_len, -1)
        st_future_actions = jnp.vstack(st_future_actions).reshape(batch_size, st_future_len, -1)

        return STermSubtrajBufferSample(
            observations=current_observations,
            actions=current_actions,
            history=History(history_observations, history_actions),
            st_future=STFuture(st_future_observations, st_future_actions)
        )

    def o_history_sample(self, batch_size: int, history_len: int, st_future_len: int = None):
        batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        timesteps = np.random.randint(0, self.traj_lengths - 1)[batch_inds].squeeze()

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        history_observations = np.zeros((batch_size, history_len, self.observation_dim))
        history_actions = np.zeros((batch_size, history_len, self.action_dim))

        st_future_observations = np.zeros((batch_size, st_future_len, self.observation_dim))
        st_future_actions = np.zeros((batch_size, st_future_len, self.action_dim))

        for idx, batch in enumerate(batch_inds):
            timestep = timesteps[idx]
            start_pts = np.max([timestep - history_len, 0])
            hist_obs = self.observation_traj[batch, start_pts: timestep, ...]
            hist_act = self.action_traj[batch, start_pts: timestep, ...]
            cur_hist_len = len(hist_obs)  # ???????????? timestep??? ????????????, ????????? ?????? history_len?????? ????????????.

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            history_observations[idx] = hist_obs
            history_actions[idx] = hist_act

            st_fut_obs = self.observation_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            st_future_observations[idx] = st_fut_obs
            st_future_actions[idx] = st_fut_act

        return STermSubtrajBufferSample(
            observations=current_observations,
            actions=current_actions,
            history=History(history_observations, history_actions),
            st_future=STFuture(st_future_observations, st_future_actions)
        )

    @staticmethod
    def timestep_marking(
        history: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        History: [batch_size, len_subtraj, obs_dim + action_dim]
        Future: [batch_size, len_subtraj, obs_dim + action_dim]
        Future may be none, especially when evaluation.

        NOTE: History, Future ?????? ?????? ???????????? ?????? ????????? ???
        Here, we add additional information that the trajectory is whether "history" or "future"

        For history --> -1, -2, -3, ...
        For future --> +1, +2, +3, ...
        """
        batch_size, len_subtraj, _ = history.shape
        history_marker = jnp.arange(-len_subtraj, 0)[None, ...] / len_subtraj
        history_marker = jnp.repeat(history_marker, repeats=batch_size, axis=0)[..., None]
        history = jnp.concatenate((history, history_marker), axis=2)

        return history


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        normalize_factor: float
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_dim=observation_dim,
            action_dim=action_dim
        )
        self.normalize_factor = normalize_factor

        self.observations = None
        self.next_observations = None
        self.actions = None
        self.rewards = None
        self.dones = None

        self.pos = None
        self.full = None

    def _normalize_obs(self, observations: np.ndarray) -> np.ndarray:
        return observations / self.normalize_factor

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.observation_dim))
        self.next_observations = np.zeros((self.buffer_size, self.observation_dim))
        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.rewards = np.zeros((self.buffer_size,))
        self.dones = np.zeros((self.buffer_size,))

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ):
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        data = (
            self._normalize_obs(self.observations[batch_inds, :]),      # Obs
            self.actions[batch_inds, :],                                # Act
            self._normalize_obs(self.next_observations[batch_inds, :]), # Next obs
            self.dones[batch_inds],
            self.rewards[batch_inds]
        )

        return ReplayBufferSamples(*data)


if __name__ == "__main__":
    import gym

    env_name = "walker2d-medium-replay-v2"
    env = gym.make(env_name)
    z = TrajectoryBuffer(f"/workspace/expertdata/dttrajectory/{env_name}",
                         env.observation_space.shape[0],
                         env.action_space.shape[0])