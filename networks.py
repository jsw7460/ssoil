from typing import List, Tuple, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from buffer import TrajectoryBuffer
from model import MLP

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


Params = flax.core.FrozenDict[str, Any]
LOG_STD_MAX = 2
LOG_STD_MIN = -10


class HistoryAutoEncoder(nn.Module):
    state_dim: int
    latent_dim: int
    squashed_output: bool = True
    dropout: float = 0.1

    encoder = None
    decoder = None
    mu = None
    log_std = None
    decoder_key = 0

    def setup(self):
        encoder_net_arch = [32, 32, self.latent_dim]
        self.encoder = MLP(net_arch=encoder_net_arch, dropout=self.dropout)

        mean_arch = [32, 32, self.latent_dim]
        self.mu = MLP(net_arch=mean_arch, dropout=self.dropout)

        log_std_arch = [32, 32, self.latent_dim]
        self.log_std = MLP(net_arch=log_std_arch, dropout=self.dropout)

        decoder_net_arch = [32, 32, self.state_dim]
        self.decoder = MLP(net_arch=decoder_net_arch, dropout=self.dropout, squashed_out=self.squashed_output)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        history: jnp.ndarray,
        deterministic: bool = False,
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[np.ndarray, ...]]:

        rng, key = self.make_rng("decoder")
        decoder_key = jax.random.PRNGKey(rng)

        if history.ndim == 1:
            history = history[np.newaxis, np.newaxis, ...]
        elif history.ndim == 2:
            history = history[np.newaxis, ...]

        history = TrajectoryBuffer.timestep_marking(history)

        mu, log_std = self.encode(history, deterministic)
        latent = self.get_latent_vector(mu, log_std, decoder_key)

        recon = self.decode(history, latent=latent, deterministic=deterministic)
        return recon, latent, (mu, log_std)

    def encode(self, history: np.ndarray, deterministic: bool):
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        """
        emb = self.encoder(history, deterministic)
        emb = np.mean(emb, axis=1)
        mu = self.mu(emb, deterministic)
        log_std = self.log_std(emb, deterministic)
        log_std = np.clip(log_std, -4.0, 10.0)

        return mu, log_std

    def decode(self, history: np.ndarray, deterministic: bool, latent: np.ndarray = None) -> jnp.ndarray:
        if latent is None:
            history, *_ = TrajectoryBuffer.timestep_marking(history)
            mu, log_std = self.encode(history, deterministic)
            latent = self.get_latent_vector(mu, log_std)

        recon = self.decoder(latent, deterministic)
        return recon

    @staticmethod
    def get_latent_vector(mu: np.ndarray, log_std: np.ndarray, key: jnp.ndarray) -> np.ndarray:
        std = jnp.exp(log_std)
        # latent = mu + std * jax.random.standard_normal(std.shape)
        latent = mu + std * jax.random.normal(key, shape=mu.shape)
        return latent


class SASPredictor(nn.Module):      # Input: s, a // Output: predicted next state (S x A --> S: SAS)
    state_dim: int
    net_arch: List = None
    dropout: float = 0.0

    predictor = None

    def setup(self):
        net_arch = self.net_arch
        if net_arch is None:
            net_arch = [256, 256, self.state_dim]
        assert net_arch[-1] == self.state_dim, "Mismatch size"
        self.predictor = MLP(net_arch, dropout=self.dropout, squashed_out=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, deterministic=False):
        return self.predictor(x, deterministic=deterministic)


class MSEActor(nn.Module):
    features_extractor: nn.Module
    action_dim: int
    net_arch: List = None
    dropout: float = 0.0

    mu = None

    def setup(self):
        net_arch = self.net_arch
        if net_arch is None:
            net_arch = [256, 256, 256, self.action_dim]
        assert net_arch[-1] == self.action_dim
        self.mu = MLP(net_arch, dropout=self.dropout, squashed_out=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, deterministic: bool = False):
        x = self.features_extractor(x)
        return self.mu(x, deterministic=deterministic)


class MLEActor(nn.Module):
    features_extractor: nn.Module
    action_dim: int
    emb_arch = None
    dec_arch = None
    dropout: float = 0.0

    latent_pi = None
    mu = None
    log_std = None

    def setup(self):
        emb_arch = self.emb_arch
        if emb_arch is None:
            emb_arch = [256, 256, 256]
        self.latent_pi = MLP(emb_arch, dropout=self.dropout, squashed_out=False)

        dec_arch = self.dec_arch
        if dec_arch is None:
            dec_arch = [256, 256, self.action_dim]
        assert dec_arch[-1] == self.action_dim
        self.mu = MLP(dec_arch, dropout=self.dropout, squashed_out=False)
        self.log_std = MLP(dec_arch, dropout=self.dropout, squashed_out=False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: jnp.ndarray, deterministic: bool = False):
        mean_actions, log_stds = self.get_action_dist_params(x, deterministic=deterministic)
        return self.actions_from_params(mean_actions, log_stds)

    def get_action_dist_params(
        self,
        x: jnp.ndarray,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = self.features_extractor(x)

        latent_pi = self.latent_pi(x, deterministic=deterministic)
        mean_actions = self.mu(latent_pi, deterministic=deterministic)
        log_stds = self.log_std(latent_pi, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_stds

    def actions_from_params(
        self,
        mean_actions: jnp.ndarray,
        log_std: jnp.ndarray,
    ) -> jnp.ndarray:
        # From mean and log std, return the actions by applying the tanh nonlinear transformation.
        rng, key = self.make_rng("action_sample")
        action_sample_key = jax.random.PRNGKey(rng)
        base_dist = tfd.MultivariateNormalDiag(loc=mean_actions, scale_diag=jnp.exp(log_std))
        sampled_action = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampled_action.sample(seed=action_sample_key)
