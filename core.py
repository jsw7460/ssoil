from typing import Dict, Any

import flax
import jax.random

from typing import Tuple

import jax.numpy as jnp
import jax.random

from buffer import STermSubtrajBufferSample
from model import Model
from networks import MLEActor, WassersteinAutoEncoder

Params = flax.core.FrozenDict[str, Any]


def update_sas(
    key: int,
    sas_predictor: Model,
    replay_data: STermSubtrajBufferSample,
):
    predictor_input = jnp.hstack((replay_data.observations, replay_data.actions))
    next_state = replay_data.st_future.observations[:, 0, :]

    def sas_predictor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        next_state_pred = sas_predictor.apply_fn(
            {"params": params},
            predictor_input,
            deterministic=False,
            rngs={"dropout": key}
        )
        sas_loss = jnp.mean((next_state_pred - next_state) ** 2)
        return sas_loss, {"sas_loss": sas_loss}

    new_actor, info = sas_predictor.apply_gradient(sas_predictor_loss_fn)
    return new_actor, info


def update_vae(
    key: jnp.ndarray,
    ae: Model,
    replay_data: STermSubtrajBufferSample,
    target_goals: jnp.ndarray,
) -> Tuple[Model, Dict]:

    vae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key)

    def vae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent, (mu, log_std) = ae.apply_fn(
            {"params": params},
            vae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key}
        )
        std = jnp.exp(log_std)
        dim = std.shape[1]

        recon_loss = jnp.mean((target_pred - target_goals) ** 2)
        kl_loss = 0.5 * (
            - jnp.log(jnp.prod(std ** 2, axis=1, keepdims=True))
            - dim
            + jnp.sum(std ** 2, axis=1, keepdims=True)
            + jnp.sum(mu ** 2, axis=1, keepdims=True)
        )
        kl_loss = jnp.mean(kl_loss)
        vae_loss = recon_loss + kl_loss
        infos = {
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mu": mu.mean(),
            "std": std.mean()
        }
        return vae_loss, infos

    new_vae, info = ae.apply_gradient(vae_loss_fn)
    return new_vae, info


def update_vae_by_policy(
    key: jnp.ndarray,
    actor: Model,
    ae: Model,
    replay_data: STermSubtrajBufferSample,

) -> Tuple[Model, Dict]:
    vae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key)

    def vae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        """
        :param params: Params of the VAE.
        """
        _, latent, *_ = ae.apply_fn(
            {"params": params},
            vae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key}
        )
        actor_input = jnp.concatenate((replay_data.observations, latent), axis=1)

        action_pred = actor.apply_fn(
            {"params": actor.params},       # This function does not update actor! It updates ae.
            actor_input,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        vae_loss = jnp.mean((action_pred - replay_data.actions) ** 2)
        return vae_loss, {"vae_loss_by_policy": vae_loss}

    new_vae, info = ae.apply_gradient(vae_loss_fn)
    return new_vae, info


def update_wae(
    key: jnp.ndarray,
    ae: Model,
    replay_data: STermSubtrajBufferSample,
    target_goals: jnp.ndarray,
) -> Tuple[Model, Dict]:
    wae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    rng, key = jax.random.split(key)
    dropout_key, mmd_key, noise_key = jax.random.split(rng, 3)

    def wae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent = ae.apply_fn(
            {"params": params},
            wae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "noise": noise_key}
        )

        mmd_loss = ae.apply_fn(
            {"params": params},
            z=latent,
            key=key,
            rngs={"dropout": dropout_key},
            method=WassersteinAutoEncoder.rbf_mmd_loss
        )

        recon_loss = jnp.mean((target_pred - target_goals) ** 2)

        wae_loss = recon_loss + mmd_loss
        infos = {
            "vae_loss": wae_loss,
            "recon_loss": recon_loss,
            "mmd_loss": mmd_loss
        }
        return wae_loss, infos

    new_wae, info = ae.apply_gradient(wae_loss_fn)
    return new_wae, info


def update_mse_actor(
    key: int,
    actor: Model,
    replay_data: STermSubtrajBufferSample,
    history_latent: jnp.ndarray,
) -> Tuple[Model, Dict]:
    actor_input = jnp.concatenate((replay_data.observations, history_latent), axis=1)

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        actor_action = actor.apply_fn(
            {"params": params},
            actor_input,
            deterministic=False,
            rngs={"dropout": key}
        )
        actor_loss = jnp.mean((actor_action - replay_data.actions) ** 2)

        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def update_mle_actor(
    key: int,
    actor: Model,
    replay_data: STermSubtrajBufferSample,
    history_latent: jnp.ndarray,
) -> Tuple[Model, Dict]:
    actor_input = jnp.concatenate((replay_data.observations, history_latent), axis=1)
    actions = replay_data.actions
    actions = actions.clip(-1.0 + 1e-5, 1.0 - 1e-5)
    gaussian_actions = jnp.arctanh(actions)

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        mu, log_std = actor.apply_fn(
            {"params": params},
            actor_input,
            deterministic=False,
            rngs={"dropout": key, "action_sample": key},
            method=MLEActor.get_action_dist_params
        )
        var = jnp.exp(log_std) ** 2

        # Compute log prob before applying the tanh transformation
        log_prob = -((gaussian_actions - mu) ** 2) / (2 * var) - log_std - jnp.log(jnp.sqrt(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=1)

        # Due to tanh, we multiply the additional part
        log_prob -= jnp.sum(jnp.log(1 - actions ** 2) + 1e-8, axis=1)

        actor_loss = -jnp.mean(log_prob)
        return actor_loss, {"actor_loss": actor_loss, "actor_mu": mu, "actor_std": jnp.exp(log_std)}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


@jax.jit
def _update_mse_jit(
    rng: int,
    actor: Model,
    ae: Model,              # Autoencoder
    sas_predictor: Model,
    replay_data: STermSubtrajBufferSample,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        replay_data=replay_data
    )
    # Get a states with highets gradient

    # Note Start: 2. VAE train
    rng, key = jax.random.split(rng)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation



    # def mean_state_change(params: Params, observations: jnp.ndarray, actions: jnp.ndarray):
    #     sas_predictor_input = jnp.concatenate((observations, actions), axis=2)
    #     next_state_pred \
    #         = sas_predictor.apply_fn({"params": params}, sas_predictor_input, rngs={"dropout": key})
    #     return jnp.mean(next_state_pred)
    #
    # diff_ft = jax.grad(mean_state_change, 2)
    # state_grads = diff_ft(sas_predictor.params, replay_data.st_future.observations, replay_data.st_future.actions)
    # state_grads = jnp.mean(state_grads, axis=2)
    # max_indices = jnp.argmax(state_grads, axis=1)
    #
    # # Now we define the target future
    # batch_size = replay_data.observations.shape[0]
    # target_goals = replay_data.st_future.observations[jnp.arange(batch_size), max_indices, ...]



    target_goals = replay_data.st_future.observations[:, 0, ...]
    new_ae, vae_info = update_vae(
        key=key,
        ae=ae,
        replay_data=replay_data,
        target_goals=target_goals
    )

    ae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    dropout_key, decoder_key, noise_key = jax.random.split(key, 3)
    _, history_latent, *_ = ae.apply_fn(
        {"params": ae.params},
        ae_input,
        deterministic=True,
        rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key},
    )

    # Note Start: 3. Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_mse_actor(
        key=key,
        actor=actor,
        replay_data=replay_data,
        history_latent=history_latent
    )

    return rng, new_sas_predictor, new_ae, new_actor, {**sas_predictor_info, **vae_info, **actor_info}


@jax.jit
def _update_mse_jit_flow(
    rng: int,
    actor: Model,
    ae: Model,              # Autoencoder
    sas_predictor: Model,
    replay_data: STermSubtrajBufferSample,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        replay_data=replay_data
    )
    # Get a states with highets gradient

    # Note Start: 2. VAE train
    rng, key = jax.random.split(rng)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    def mean_state_change(params: Params, observations: jnp.ndarray, actions: jnp.ndarray):
        sas_predictor_input = jnp.concatenate((observations, actions), axis=2)
        next_state_pred \
            = sas_predictor.apply_fn({"params": params}, sas_predictor_input, rngs={"dropout": key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 2)
    state_grads = diff_ft(sas_predictor.params, replay_data.st_future.observations, replay_data.st_future.actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = replay_data.observations.shape[0]
    target_goals = replay_data.st_future.observations[jnp.arange(batch_size), max_indices, ...]
    new_ae, vae_info = update_vae(
        key=key,
        ae=ae,
        replay_data=replay_data,
        target_goals=target_goals
    )

    ae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key, 2)
    _, history_latent, *_ = ae.apply_fn(
        {"params": ae.params},
        ae_input,
        deterministic=True,
        rngs={"dropout": dropout_key, "decoder": decoder_key},
    )

    # Note Start: 3. Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_mse_actor(
        key=key,
        actor=actor,
        replay_data=replay_data,
        history_latent=history_latent
    )

    rng, key = jax.random.split(rng)
    new_ae, vae_flow_info = update_vae_by_policy(
        key=key,
        actor=actor,
        ae=new_ae,
        replay_data=replay_data
    )

    return rng, new_sas_predictor, new_ae, new_actor, {**sas_predictor_info, **vae_info, **actor_info, **vae_flow_info}


@jax.jit
def _update_mle_jit(
        rng: int,
        actor: Model,
        vae: Model,
        sas_predictor: Model,
        replay_data: STermSubtrajBufferSample,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        replay_data=replay_data
    )
    # Get a states with highets gradient

    # Note Start: 2. VAE train
    rng, key = jax.random.split(rng)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    def mean_state_change(params: Params, observations: jnp.ndarray, actions: jnp.ndarray):
        sas_predictor_input = jnp.concatenate((observations, actions), axis=2)
        next_state_pred \
            = sas_predictor.apply_fn({"params": params}, sas_predictor_input, rngs={"dropout": key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 2)
    state_grads = diff_ft(sas_predictor.params, replay_data.st_future.observations, replay_data.st_future.actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = replay_data.observations.shape[0]
    target_goals = replay_data.st_future.observations[jnp.arange(batch_size), max_indices, ...]

    new_vae, vae_info = update_vae(
        key=key,
        vae=vae,
        replay_data=replay_data,
        target_goals=target_goals
    )

    vae_input = jnp.concatenate((replay_data.history.observations, replay_data.history.actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key, 2)
    _, history_latent, *_ = vae.apply_fn(
        {"params": vae.params},
        vae_input,
        deterministic=True,
        rngs={"dropout": dropout_key, "decoder": decoder_key}
    )

    # Note Start: 3. Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_mle_actor(
        key=key,
        actor=actor,
        replay_data=replay_data,
        history_latent=history_latent
    )
    return rng, new_sas_predictor, new_vae, new_actor, {**sas_predictor_info, **vae_info, **actor_info}