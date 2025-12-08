from __future__ import annotations

import typing as tp
from types import MappingProxyType

from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module
from flax.nnx.nn import initializers, dtypes
from flax.nnx.nn.linear import Linear
from flax.typing import Dtype, Initializer, PromoteDtypeFn
import jax
import jax.numpy as jnp

Array = jax.Array
Axis = int
Size = int
A = tp.TypeVar('A')

default_a_initializer = initializers.he_uniform()
default_b_initializer = initializers.zeros


class DoRAParam(variablelib.Param[A]):
    pass


class DoRA(Module):
    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        *,
        base_module: tp.Optional[Module] = None,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        a_initializer: Initializer = default_a_initializer,
        b_initializer: Initializer = default_b_initializer,
        magnitude_initializer: Initializer = initializers.ones,
        dora_param_type: tp.Type[variablelib.Variable] = DoRAParam,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
        a_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        b_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        magnitude_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        eps: float = 1e-6,
        ):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.dora_param_type = dora_param_type
        self.base_module = base_module
        self.promote_dtype = promote_dtype
        self.eps = eps

        self.lora_a = dora_param_type(
            a_initializer(rngs.params(), (in_features, lora_rank), param_dtype),
            **a_metadata,
        )
        self.lora_b = dora_param_type(
            b_initializer(rngs.params(), (lora_rank, out_features), param_dtype),
            **b_metadata,
        )

        if base_module is not None and hasattr(base_module, 'kernel'):
            base_weight = base_module.kernel.value
            initial_magnitude = jnp.linalg.norm(base_weight, axis=0, keepdims=False)
            self.magnitude = dora_param_type(
            initial_magnitude.astype(param_dtype),
            **magnitude_metadata,
            )
        else:
            self.magnitude = dora_param_type(
            magnitude_initializer(rngs.params(), (out_features,), param_dtype),
            **magnitude_metadata,
            )

    def __call__(self, x: jax.Array):
        x, lora_a, lora_b, magnitude = self.promote_dtype(
            (x, self.lora_a[...], self.lora_b[...], self.magnitude[...]), 
            dtype=self.dtype
        )
        lora_update = lora_a @ lora_b
        
        if self.base_module is None:
            raise ValueError("DoRA requires `base_module` to be provided.")
        
        if not callable(self.base_module):
            raise ValueError('`self.base_module` must be callable.')
        
        if not hasattr(self.base_module, 'kernel'):
            raise ValueError('`self.base_module` must have a `kernel` attribute.')
        
        base_weight = self.base_module.kernel.value
        base_weight = jnp.asarray(base_weight, dtype=self.dtype)
        
        # DoRA: decompose base weight into magnitude and direction
        weight_norm = jnp.linalg.norm(base_weight, axis=0, keepdims=True)
        weight_norm = jnp.maximum(weight_norm, self.eps)  
        directional_base = base_weight / weight_norm
        scaled_base = magnitude * directional_base
        adapted_weight = scaled_base + lora_update
        
        out = x @ adapted_weight
        
        if hasattr(self.base_module, 'bias') and self.base_module.bias is not None:
            bias = jnp.asarray(self.base_module.bias.value, dtype=self.dtype)
            out += bias
        
        return out


class DoRALinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lora_rank: int,
        dora_dtype: tp.Optional[Dtype] = None,
        dora_param_dtype: Dtype = jnp.float32,
        a_initializer: Initializer = default_a_initializer,
        b_initializer: Initializer = default_b_initializer,
        magnitude_initializer: Initializer = initializers.ones,
        dora_param_type: tp.Type[variablelib.Variable] = DoRAParam,
        dora_promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
        a_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        b_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        magnitude_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(in_features, out_features, rngs=rngs, **kwargs)

        self.dora = DoRA(
            in_features,
            lora_rank,
            out_features,
            base_module=self,
            dtype=dora_dtype,
            param_dtype=dora_param_dtype,
            a_initializer=a_initializer,
            b_initializer=b_initializer,
            magnitude_initializer=magnitude_initializer,
            dora_param_type=dora_param_type,
            promote_dtype=dora_promote_dtype,
            rngs=rngs,
            a_metadata=a_metadata,
            b_metadata=b_metadata,
            magnitude_metadata=magnitude_metadata,
            eps=eps,
        )

    def __call__(self, x: jax.Array):
        return self.dora(x)
