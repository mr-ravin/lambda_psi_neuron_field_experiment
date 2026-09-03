"""Model definitions for the Lambda-Psi comparison study.

This module contains only architecture-related code:
- the ten experimental variant specifications,
- construction of Traditional or APTx base neuron layers,
- explicit ReLU/Lambda-Psi ordering inside hidden blocks, and
- the complete three-hidden-layer comparison network.

The Lambda-Psi equation itself remains in ``lambda_psi_neuron_field`` so the
benchmark and the reusable package execute the same mathematical definition.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from lambda_psi_neuron_field import aptx_neuron_layer, lambda_psi_field_layer

# -----------------------------------------------------------------------------
# Experiment variants
# -----------------------------------------------------------------------------
# Each specification states which base neuron is used, whether Lambda-Psi is
# present, and where ReLU is positioned. This is the single source of truth for
# all ten requested architectural comparisons.
VARIANT_SPECS: Dict[str, Dict[str, object]] = {
    "traditional": {
        "label": "Traditional neuron",
        "base": "traditional",
        "use_field": False,
        "relu_position": "none",
    },
    "traditional_relu": {
        "label": "Traditional neuron + ReLU",
        "base": "traditional",
        "use_field": False,
        "relu_position": "after_neuron",
    },
    "traditional_field": {
        "label": "Traditional neuron + Lambda-Psi",
        "base": "traditional",
        "use_field": True,
        "relu_position": "none",
    },
    "traditional_field_relu": {
        "label": "Traditional neuron + Lambda-Psi + ReLU",
        "base": "traditional",
        "use_field": True,
        "relu_position": "after_field",
    },
    "traditional_relu_field": {
        "label": "Traditional neuron + ReLU + Lambda-Psi",
        "base": "traditional",
        "use_field": True,
        "relu_position": "before_field",
    },
    "aptx": {
        "label": "APTx Neuron",
        "base": "aptx",
        "use_field": False,
        "relu_position": "none",
    },
    "aptx_relu": {
        "label": "APTx Neuron + ReLU",
        "base": "aptx",
        "use_field": False,
        "relu_position": "after_neuron",
    },
    "aptx_field": {
        "label": "APTx Neuron + Lambda-Psi",
        "base": "aptx",
        "use_field": True,
        "relu_position": "none",
    },
    "aptx_field_relu": {
        "label": "APTx Neuron + Lambda-Psi + ReLU",
        "base": "aptx",
        "use_field": True,
        "relu_position": "after_field",
    },
    "aptx_relu_field": {
        "label": "APTx Neuron + ReLU + Lambda-Psi",
        "base": "aptx",
        "use_field": True,
        "relu_position": "before_field",
    },
}

# -----------------------------------------------------------------------------
# Lambda-Psi field alias
# -----------------------------------------------------------------------------
# The field implementation lives in the package; the alias keeps model code short.
LambdaPsiField = lambda_psi_field_layer

# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
def build_base_neuron(
    base: str,
    input_dim: int,
    output_dim: int,
    is_alpha_trainable: bool = True,
    use_delta: bool = True,
) -> nn.Module:
    if base == "traditional":
        return nn.Linear(input_dim, output_dim, bias=use_delta)
    if base == "aptx":
        return aptx_neuron_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            is_alpha_trainable=is_alpha_trainable,
            use_delta=use_delta,
        )
    raise ValueError(f"Unsupported base neuron: {base}")

class HiddenBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base: str,
        use_field: bool,
        relu_position: str,
        lambda_init: float,
        psi_init: float,
        is_alpha_trainable: bool = True,
        use_delta: bool = True,
    ) -> None:
        super().__init__()
        self.relu_position = relu_position
        self.neuron = build_base_neuron(
            base=base,
            input_dim=input_dim,
            output_dim=output_dim,
            is_alpha_trainable=is_alpha_trainable,
            use_delta=use_delta,
        )
        self.field = (
            LambdaPsiField(lambda_init=lambda_init, psi_init=psi_init)
            if use_field
            else None
        )

        valid = {"none", "after_neuron", "before_field", "after_field"}
        if relu_position not in valid:
            raise ValueError(f"Invalid relu_position={relu_position!r}")
        if relu_position in {"before_field", "after_field"} and self.field is None:
            raise ValueError(
                f"relu_position={relu_position!r} requires use_field=True"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.neuron(x)

        if self.relu_position == "after_neuron":
            return torch.relu(y)

        if self.relu_position == "before_field":
            y = torch.relu(y)
            return self.field(y)

        if self.field is not None:
            y = self.field(y)

        if self.relu_position == "after_field":
            y = torch.relu(y)

        return y

class ComparisonNetwork(nn.Module):
    """Three hidden blocks plus a base-neuron output head by default.

    The paper defines Lambda-Psi as a layer-level operator, not as a requirement that
    every network layer must be a field layer.  The primary comparison therefore
    applies the requested ordering to hidden representation layers and leaves the
    task head as the corresponding base neuron.  ``field_on_output=True`` is retained
    as an explicit ablation for experiments that also make the output head a field.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, int, int],
        variant: str,
        lambda_init: float = 0.5,
        psi_init: float = 0.5,
        field_on_output: bool = False,
        is_alpha_trainable: bool = True,
        use_delta: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant={variant!r}")

        spec = VARIANT_SPECS[variant]
        base = str(spec["base"])
        use_field = bool(spec["use_field"])
        relu_position = str(spec["relu_position"])

        h1, h2, h3 = hidden_dims
        dims = [(input_dim, h1), (h1, h2), (h2, h3)]

        self.variant = variant
        self.base = base
        self.use_field = use_field
        self.field_on_output = field_on_output and use_field

        self.blocks = nn.ModuleList(
            [
                HiddenBlock(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    base=base,
                    use_field=use_field,
                    relu_position=relu_position,
                    lambda_init=lambda_init,
                    psi_init=psi_init,
                    is_alpha_trainable=is_alpha_trainable,
                    use_delta=use_delta,
                )
                for in_dim, out_dim in dims
            ]
        )

        self.output_neuron = build_base_neuron(
            base=base,
            input_dim=h3,
            output_dim=output_dim,
            is_alpha_trainable=is_alpha_trainable,
            use_delta=use_delta,
        )
        self.output_field = (
            LambdaPsiField(lambda_init=lambda_init, psi_init=psi_init)
            if self.field_on_output
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for block in self.blocks:
            x = block(x)
        out = self.output_neuron(x)
        if self.output_field is not None:
            out = self.output_field(out)
        return out

    def field_values(self) -> Dict[str, Tuple[float, float]]:
        values: Dict[str, Tuple[float, float]] = {}
        for index, block in enumerate(self.blocks, start=1):
            if block.field is not None:
                values[f"field{index}"] = (
                    float(block.field.lambda_value.detach().cpu()),
                    float(block.field.psi_value.detach().cpu()),
                )
        if self.output_field is not None:
            values["output"] = (
                float(self.output_field.lambda_value.detach().cpu()),
                float(self.output_field.psi_value.detach().cpu()),
            )
        return values
