import os
from typing import Callable, Any, Optional, Union, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import struct

Params = flax.core.FrozenDict[str, Any]


@struct.dataclass
class Model(object):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None
    ) -> 'Model':

        variables = model_def.init(*inputs)
        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1, apply_fn=model_def.apply, params=params, tx=tx, opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True
    ) -> Union[Tuple['Model', Any], 'Model']:

        assert (loss_fn is not None or grads is not None, 'Either a loss function or grads must be specified.')

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux, 'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save_dict(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))
        return self.params

    def load_dict(self, params: bytes) -> 'Model':
        params = flax.serialization.from_bytes(self.params, params)
        return self.replace(params=params)
