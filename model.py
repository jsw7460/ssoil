import os
from typing import Callable, Union, Sequence
from typing import List, Optional, Tuple, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

Params = flax.core.FrozenDict[str, Any]


def default_init():
    return nn.initializers.he_normal()


class MLP(nn.Module):
    net_arch: List
    dropout: float
    squashed_out: bool = False

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for feature in self.net_arch[:-1]:
            x = nn.relu(nn.Dense(feature, kernel_init=default_init())(x))
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.net_arch[-1], kernel_init=default_init())(x)
        if self.squashed_out:
            return nn.tanh(x)
        else:
            return x


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
