import numpy as np
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from abc import ABC

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

    def sample(self, batch_size: int):
        pass


class TrajectoryBuffer(BaseBuffer):
    def __init__(
        self,
        data_path: str,
        observation_dim: int,
        action_dim: int,
        normalize: bool = True,
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

        self.expert_dataset = expert_dataset
        self.max_traj_len = max_traj_len
        self.normalize = normalize
        self.normalizing_factor = None

        self.observation_traj = None
        self.action_traj = None
        self.reward_traj = None
        self.terminal_traj = None
        self.traj_lengths = None

        self._reset()

    def _reset(self):
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

        self.observation_traj /= self.normalizing_factor

    def history_sample(self, batch_size: int, history_len: int, st_future_len: int = None):
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
            hist_obs = self.observation_traj[batch, timestep - history_len: timestep, ...]
            hist_act = self.action_traj[batch, timestep - history_len: timestep, ...]
            cur_hist_len = len(hist_obs)        # 앞쪽에서 timestep이 골라지면, history_len보다 짧아진다.

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

        NOTE: History, Future 표현 방식 바꾸려면 여기 바꿔야 함
        Here, we add additional information that the trajectory is whether "history" or "future"

        For history --> -1, -2, -3, ...
        For future --> +1, +2, +3, ...
        """
        batch_size, len_subtraj, _ = history.shape
        history_marker = jnp.arange(-len_subtraj, 0)[None, ...] / len_subtraj
        history_marker = jnp.repeat(history_marker, repeats=batch_size, axis=0)[..., None]
        history = jnp.concatenate((history, history_marker), axis=2)

        return history


if __name__ == "__main__":
    import gym
    import d4rl
    env_name = "walker2d-medium-replay-v2"
    env = gym.make(env_name)
    z = TrajectoryBuffer(f"/workspace/expertdata/dttrajectory/{env_name}",
                         env.observation_space.shape[0],
                         env.action_space.shape[0])