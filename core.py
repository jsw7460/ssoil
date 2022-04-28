from typing import Dict, Any
from typing import Tuple

import flax
import jax.numpy as jnp
import jax.random
import jax.random

from buffer import StateActionBufferSample
from core_comp import (
    update_vae,
    update_sas,
    update_vae_by_mle_policy,
    update_mle_actor,
    update_mse_actor,
    update_vae_by_mse_policy,
    update_mse_actor_by_goal_loss
)
from model import Model

Params = flax.core.FrozenDict[str, Any]


@jax.jit
def update_bc(
    key: jnp.ndarray,
    actor: Model,
    replay_data: StateActionBufferSample
):
    noise = jax.random.normal(key, shape=replay_data.observations.shape,)
    actor_input = jnp.concatenate((replay_data.observations, noise), axis=1)

    def bc_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict]:
        action_pred = actor.apply_fn(
            {"params": params},
            actor_input,
            deterministic=True,
            rngs={"dropout": key}
        )
        bc_loss = (action_pred - replay_data.actions) ** 2
        bc_loss = jnp.mean(bc_loss)

        return bc_loss, {"bc_loss": bc_loss}

    new_actor, info = actor.apply_gradient(bc_loss_fn)
    return new_actor, info


@jax.jit
def _update_mse_jit(
        rng: jnp.ndarray,
        actor: Model,
        ae: Model,
        sas_predictor: Model,
        history_observations: jnp.ndarray,
        history_actions: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        goal_observations: jnp.ndarray,
        goal_actions: jnp.ndarray,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        observations=observations,
        actions=actions,
        next_state=next_observations
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
    state_grads = diff_ft(sas_predictor.params, goal_observations, goal_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = observations.shape[0]
    target_goals = goal_observations[jnp.arange(batch_size), max_indices, ...]

    new_ae, vae_info = update_vae(
        key=key,
        ae=ae,
        history_observations=history_observations,
        history_actions=history_actions,
        target_goals=target_goals,
    )

    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
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
        observations=observations,
        actions=actions,
        history_latent=history_latent
    )

    return rng, new_sas_predictor, new_ae, new_actor, {
        **sas_predictor_info,
        **vae_info,
        **actor_info,
        "target_goals": target_goals
    }


@jax.jit
def _update_mse_jit_flow(
        rng: int,
        actor: Model,
        ae: Model,
        sas_predictor: Model,
        history_observations: jnp.ndarray,
        history_actions: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        goal_observations: jnp.ndarray,
        goal_actions: jnp.ndarray,
):
    # 1. SAS train 2. VAE train 3. Actor train
    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        observations=observations,
        actions=actions,
        next_state=next_observations,
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
    state_grads = diff_ft(sas_predictor.params, goal_observations, goal_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = observations.shape[0]
    target_goals = goal_observations[jnp.arange(batch_size), max_indices, ...]
    new_ae, vae_info = update_vae(
        key=key,
        ae=ae,
        history_observations=history_observations,
        history_actions=history_actions,
        target_goals=target_goals
    )

    ae_input = jnp.concatenate((history_observations, history_actions), axis=2)
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
        observations=observations,
        actions=actions,
        history_latent=history_latent
    )

    # Note Start: 4. VAE train by gradients of actor
    rng, key = jax.random.split(rng)
    new_ae, vae_flow_info = update_vae_by_mse_policy(
        key=key,
        actor=new_actor,
        ae=new_ae,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
    )

    return rng, new_sas_predictor, new_ae, new_actor, {**sas_predictor_info, **vae_info, **actor_info, **vae_flow_info}


@jax.jit
def _update_mle_jit(
        rng: int,
        actor: Model,
        ae: Model,
        sas_predictor: Model,
        history_observations: jnp.ndarray,
        history_actions: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        goal_observations: jnp.ndarray,
        goal_actions: jnp.ndarray,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        observations=observations,
        actions=actions,
        next_state=next_observations
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
    state_grads = diff_ft(sas_predictor.params, goal_observations, goal_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = observations.shape[0]
    target_goals = goal_observations[jnp.arange(batch_size), max_indices, ...]

    new_vae, vae_info = update_vae(
        key=key,
        ae=ae,
        history_observations=history_observations,
        history_actions=history_actions,
        target_goals=target_goals,
    )

    vae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key, 2)
    _, history_latent, *_ = ae.apply_fn(
        {"params": ae.params},
        vae_input,
        deterministic=True,
        rngs={"dropout": dropout_key, "decoder": decoder_key}
    )

    # Note Start: 3. Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_mle_actor(
        key=key,
        actor=actor,
        observations=observations,
        actions=actions,
        history_latent=history_latent
    )
    return rng, new_sas_predictor, new_vae, new_actor, {**sas_predictor_info, **vae_info, **actor_info}


@jax.jit
def _update_mle_jit_flow(
    rng: int,
    actor: Model,
    ae: Model,
    sas_predictor: Model,
    history_observations: jnp.ndarray,
    history_actions: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    goal_observations: jnp.ndarray,
    goal_actions: jnp.ndarray,
):
    # 1. SAS train 2. VAE train 3. Actor train

    # Note Start: 1. SAS train
    rng, key = jax.random.split(rng)
    new_sas_predictor, sas_predictor_info = update_sas(
        key=key,
        sas_predictor=sas_predictor,
        observations=observations,
        actions=actions,
        next_state=next_observations,
    )
    # Get a states with highets gradient

    # Note Start: 2. VAE train
    rng, key = jax.random.split(rng)

    # Compute the maximum gradient state
    # To do this, we have to define the function to calculate the partial derivation
    def mean_state_change(params: Params, _observations: jnp.ndarray, _actions: jnp.ndarray):
        sas_predictor_input = jnp.concatenate((_observations, _actions), axis=2)
        next_state_pred \
            = sas_predictor.apply_fn({"params": params}, sas_predictor_input, rngs={"dropout": key})
        return jnp.mean(next_state_pred)

    diff_ft = jax.grad(mean_state_change, 2)
    state_grads = diff_ft(sas_predictor.params, goal_observations, goal_actions)
    state_grads = jnp.mean(state_grads, axis=2)
    max_indices = jnp.argmax(state_grads, axis=1)

    # Now we define the target future
    batch_size = observations.shape[0]
    target_goals = goal_observations[jnp.arange(batch_size), max_indices, ...]

    new_ae, vae_info = update_vae(
        key=key,
        ae=ae,
        history_observations=history_observations,
        history_actions=history_actions,
        target_goals=target_goals
    )

    vae_input = jnp.concatenate((history_observations, history_actions), axis=2)
    dropout_key, decoder_key = jax.random.split(key, 2)
    _, history_latent, *_ = ae.apply_fn(
        {"params": ae.params},
        vae_input,
        deterministic=True,
        rngs={"dropout": dropout_key, "decoder": decoder_key}
    )

    # Note Start: 3. Actor train
    rng, key = jax.random.split(rng)

    new_actor, actor_info = update_mle_actor(
        key=key,
        actor=actor,
        observations=observations,
        actions=actions,
        history_latent=history_latent
    )

    # Note Start: 4. VAE train by gradients of actor
    rng, key = jax.random.split(rng)
    new_ae, vae_flow_info = update_vae_by_mle_policy(
        key=key,
        actor=new_actor,
        ae=new_ae,
        observations=observations,
        actions=actions,
        history_observations=history_observations,
        history_actions=history_actions
    )
    return rng, new_sas_predictor, new_ae, new_actor, {**sas_predictor_info, **vae_info, **actor_info, **vae_flow_info}


@jax.jit
def _update_mse_jit_goal_loss(
        rng: jnp.ndarray,
        actor: Model,
        ae: Model,
        sas_predictor: Model,
        history_observations: jnp.ndarray,
        history_actions: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        goal_observations: jnp.ndarray,
        goal_actions: jnp.ndarray,
):
    rng, new_sas_predictor, new_ae, new_actor, info = _update_mse_jit(
        rng=rng,
        actor=actor,
        ae=ae,
        sas_predictor=sas_predictor,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        goal_observations=goal_observations,
        goal_actions=goal_actions
    )

    # target_goals = info["target_goals"]
    target_goals = goal_observations
    new_actor, goal_info = update_mse_actor_by_goal_loss(
        key=rng,
        sas_predictor=new_sas_predictor,
        ae=new_ae,
        actor=new_actor,
        history_observations=history_observations,
        history_actions=history_actions,
        observations=observations,
        target_goals=target_goals,
    )
    return rng, new_sas_predictor, new_ae, new_actor, {**info, **goal_info}