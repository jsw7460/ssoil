from typing import Dict, Any
from typing import Tuple

import flax
import jax.numpy as jnp
import jax.random
import jax.random

from model import Model
from networks import MLEActor, WassersteinAutoEncoder

Params = flax.core.FrozenDict[str, Any]


def update_sas(
    key: int,
    sas_predictor: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_state: jnp.ndarray,
):
    predictor_input = jnp.hstack((observations, actions))
    # predictor_input = jnp.hstack((replay_data.observations, replay_data.actions))
    # next_state = replay_data.st_future.observations[:, 0, :]

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
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    target_goals: jnp.ndarray,
) -> Tuple[Model, Dict]:
    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    dropout_key, decoder_key, noise_key = jax.random.split(key, 3)

    def ae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent, (mu, log_std) = ae.apply_fn(
            {"params": params},
            ae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
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
        ae_loss = recon_loss + kl_loss
        infos = {
            "ae_loss": ae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mu": mu.mean(),
            "std": std.mean()
        }
        return ae_loss, infos

    new_ae, info = ae.apply_gradient(ae_loss_fn)
    return new_ae, info


def update_vae_by_mse_policy(
    key: jnp.ndarray,
    actor: Model,
    ae: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,

) -> Tuple[Model, Dict]:
    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    dropout_key, decoder_key, noise_key = jax.random.split(key, 3)

    def vae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        """
        :param params: Params of the VAE.
        """
        _, latent, *_ = ae.apply_fn(
            {"params": params},
            ae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
        )
        actor_input = jnp.concatenate((observations, latent), axis=1)

        action_pred = actor.apply_fn(
            {"params": actor.params},       # This function does not update actor! It updates ae.
            actor_input,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        vae_loss = jnp.mean((action_pred - actions) ** 2)
        return vae_loss, {"vae_loss_by_policy": vae_loss}

    new_vae, info = ae.apply_gradient(vae_loss_fn)
    return new_vae, info


def update_vae_by_mle_policy(
    key: jnp.ndarray,
    actor: Model,
    ae: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,

) -> Tuple[Model, Dict]:
    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key)

    actions = actions.clip(-1.0 + 1e-5, 1.0 - 1e-5)
    gaussian_actions = jnp.arctanh(actions)

    def vae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        """
        :param params: Params of the VAE.
        """
        _, latent, *_ = ae.apply_fn(
            {"params": params},
            ae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key}
        )

        actor_input = jnp.concatenate((observations, latent), axis=1)
        mu, log_std = actor.apply_fn(
            {"params": actor.params},
            actor_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "action_sample": decoder_key},
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

    new_vae, info = ae.apply_gradient(vae_loss_fn)
    return new_vae, info


def update_wae(
    key: jnp.ndarray,
    ae: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    target_goals: jnp.ndarray,
) -> Tuple[Model, Dict]:

    wae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    rng, key = jax.random.split(key)
    dropout_key, mmd_key, decoder_key, noise_key = jax.random.split(rng, 4)

    def wae_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        target_pred, latent = ae.apply_fn(
            {"params": params},
            wae_input,
            deterministic=False,
            rngs={"dropout": dropout_key, "decoder": decoder_key, "noise": noise_key}
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
    key: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    history_latent: jnp.ndarray,
) -> Tuple[Model, Dict]:
    actor_input = jnp.concatenate((observations, history_latent), axis=1)

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        actor_action = actor.apply_fn(
            {"params": params},
            actor_input,
            deterministic=False,
            rngs={"dropout": key}
        )
        actor_loss = jnp.mean((actor_action - actions) ** 2)

        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def update_mle_actor(
    key: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    history_latent: jnp.ndarray,
) -> Tuple[Model, Dict]:
    actor_input = jnp.concatenate((observations, history_latent), axis=1)
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
        return actor_loss, {"actor_loss": actor_loss, "actor_mu": jnp.mean(mu), "actor_log_std": jnp.mean(log_std)}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def update_mse_actor_by_goal_loss(
    key: jnp.ndarray,
    sas_predictor: Model,
    ae: Model,
    actor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    target_goals: jnp.ndarray,
):
    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    actor_key, sas_key, noise_key = jax.random.split(key, 3)
    _, latent, *_ = ae.apply_fn(
        {"params": ae.params},
        ae_input,
        deterministic=False,
        rngs={"dropout": actor_key, "decoder": actor_key, "noise": noise_key}
    )
    policy_input = jnp.concatenate((observations, latent), axis=1)

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        action_pred = actor.apply_fn(
            {"params": params},
            policy_input,
            deterministic=False,
            rngs={"dropout": actor_key}
        )
        predictor_input = jnp.concatenate((observations, action_pred), axis=1)

        next_state_pred = sas_predictor.apply_fn(
            {"params": sas_predictor.params},
            predictor_input,
            deterministic=False,
            rngs={"dropout": sas_key}
        )
        goal_loss = jnp.mean((next_state_pred - target_goals) ** 2) * 0.5

        return goal_loss, {"goal_loss": goal_loss}

    new_actor, goal_info = actor.apply_gradient(actor_loss_fn)
    return new_actor, goal_info
