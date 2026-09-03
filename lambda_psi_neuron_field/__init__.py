"""Lambda-Psi Neuron Field Layers and APTx neuron primitives.

The Lambda-Psi implementation follows the paper's two-stage formulation:

    x -> y_j -> m_j

The APTx activation/neuron/layer code below is preserved from the supplied package
implementation.  The field is neuron-agnostic and ReLU is intentionally external so
activation-order ablations remain explicit.
"""

import torch
import torch.nn as nn

__version__ = '0.0.1'

__all__ = [
    "aptx_activation_function",
    "aptx_neuron",
    "aptx_neuron_layer",
    "lambda_psi_field_layer",
    "traditional_neuron_lambda_psi_field_layer",
    "aptx_neuron_lambda_psi_field_layer",
]


# -----------------------------------
# APTx Activation Function
# -----------------------------------
class aptx_activation_function(nn.Module):
    r"""The APTx Activation Function (Alpha Plus Tanh Times): 
    Research Paper:: APTx: Better Activation Function than MISH, SWISH, and ReLU's Variants used in Deep Learning
    DOI Link: https://doi.org/10.51483/IJAIML.2.2.2022.56-61
    Arxiv: https://arxiv.org/abs/2209.06119
    
    .. math::
        \mathrm{APTxActivationFunction}(x) = (\alpha + \tanh(\beta x)) \cdot \gamma x
    :param alpha: Initial α value (default: 1.0)
    :param beta: Initial β value (default: 1.0)
    :param gamma: Initial γ value (default: 0.5)
    :param trainable: If True,  β, γ parameters become learnable (default: False)
    :param is_alpha_trainable: If True, α parameter become learnable (default: False)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, trainable=False, is_alpha_trainable=False):
        super().__init__()
        # Convert to tensors first
        alpha = torch.as_tensor(float(alpha))
        beta = torch.as_tensor(float(beta))
        gamma = torch.as_tensor(float(gamma))
        if trainable:
            if is_alpha_trainable:
                self.alpha = nn.Parameter(alpha)
            else:
                self.register_buffer("alpha", alpha)
            self.beta = nn.Parameter(beta)
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("alpha", alpha)
            self.register_buffer("beta", beta)
            self.register_buffer("gamma", gamma)

    def forward(self, x):
        """Forward pass"""
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x


# -----------------------------------
# APTx Neuron
# -----------------------------------
class aptx_neuron(nn.Module):
    def __init__(self, input_dim, is_alpha_trainable=True, use_delta=True):
        super().__init__()
        self.use_delta = use_delta
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(input_dim))
        else:
            self.register_buffer('alpha', torch.ones(input_dim))
        self.beta = nn.Parameter(torch.randn(input_dim))
        self.gamma = nn.Parameter(torch.randn(input_dim))
        if self.use_delta:
            self.delta = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("delta", None)

    def forward(self, x):  # x: [batch_size, input_dim]
        nonlinear = (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x
        # [batch_size, output_dim]
        y = nonlinear.sum(dim=1, keepdim=True)
        if self.use_delta:
            y = y + self.delta
        return y


# -----------------------------------
# APTx Layer (Vectorized Multiple Neurons)
# -----------------------------------
class aptx_neuron_layer(nn.Module):
    def __init__(self, input_dim, output_dim, is_alpha_trainable=True, use_delta=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_delta = use_delta
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(output_dim, input_dim))
        else:
            self.register_buffer('alpha', torch.ones(output_dim, input_dim))
        self.beta = nn.Parameter(torch.randn(output_dim, input_dim))
        self.gamma = nn.Parameter(torch.randn(output_dim, input_dim))
        if self.use_delta:
            self.delta = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("delta", None)

    def forward(self, x):  # x: [batch_size, input_dim]
        # x -> [batch_size, 1, input_dim]
        x_exp = x.unsqueeze(1)
        nonlinear = (
            self.alpha + torch.tanh(self.beta.unsqueeze(0) * x_exp)
        ) * self.gamma.unsqueeze(0) * x_exp
        # [batch_size, output_dim]
        y = nonlinear.sum(dim=2)
        if self.use_delta:
            y = y + self.delta
        return y

# -----------------------------------
# Generic Lambda-Psi Field Layer
# -----------------------------------
class lambda_psi_field_layer(nn.Module):
    r"""Neuron-agnostic Lambda-Psi field operating on pre-field outputs ``y``.

    This is the direct implementation of the paper's field equation:

    .. math::
        m_j = \lambda[(1-\psi)\sigma(y_j) + \psi\,\mathrm{softmax}(y)_j]
              + (1-\lambda)[(1-\psi)y_j + \psi\,\bar{y}]

    ``lambda`` controls the nature of the field and ``psi`` controls its scope.
    Both are bounded in [0, 1].  When trainable, raw scalar parameters are passed
    through sigmoid, matching the bounded parameterization suggested in the paper.

    ReLU is deliberately *not* part of this class.  This keeps the field true to
    the paper and makes ``field -> ReLU`` and ``ReLU -> field`` distinct experiments.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        psi_init: float = 0.5,
        trainable_field: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.trainable_field = bool(trainable_field)
        self.eps = float(eps)

        self._validate_unit_interval("lambda_init", lambda_init)
        self._validate_unit_interval("psi_init", psi_init)

        if self.trainable_field:
            # Exact 0/1 would correspond to +/- infinity in raw-logit space and
            # would not be a useful trainable initialization.  Clamp only for the
            # trainable representation; fixed fields below preserve exact boundaries.
            lambda_t = torch.tensor(float(lambda_init), dtype=torch.float32).clamp(
                self.eps, 1.0 - self.eps
            )
            psi_t = torch.tensor(float(psi_init), dtype=torch.float32).clamp(
                self.eps, 1.0 - self.eps
            )
            self.lambda_raw = nn.Parameter(torch.logit(lambda_t))
            self.psi_raw = nn.Parameter(torch.logit(psi_t))
            self.register_buffer("lambda_fixed", None)
            self.register_buffer("psi_fixed", None)
        else:
            self.register_parameter("lambda_raw", None)
            self.register_parameter("psi_raw", None)
            self.register_buffer(
                "lambda_fixed", torch.tensor(float(lambda_init), dtype=torch.float32)
            )
            self.register_buffer(
                "psi_fixed", torch.tensor(float(psi_init), dtype=torch.float32)
            )

    @staticmethod
    def _validate_unit_interval(name: str, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def lambda_value(self) -> torch.Tensor:
        if self.lambda_raw is None:
            return self.lambda_fixed
        return torch.sigmoid(self.lambda_raw)

    @property
    def psi_value(self) -> torch.Tensor:
        if self.psi_raw is None:
            return self.psi_fixed
        return torch.sigmoid(self.psi_raw)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2:
            raise ValueError(
                "lambda_psi_field_layer expects y with shape "
                "[batch_size, number_of_neurons]"
            )

        lam = self.lambda_value.to(dtype=y.dtype, device=y.device)
        psi = self.psi_value.to(dtype=y.dtype, device=y.device)

        # Probabilistic/distributional branch:
        # lambda=1, psi=0 -> Sigmoid
        # lambda=1, psi=1 -> Softmax
        sigmoid_y = torch.sigmoid(y)
        softmax_y = torch.softmax(y, dim=1)
        probabilistic_branch = (1.0 - psi) * sigmoid_y + psi * softmax_y

        # Linear/contextual branch:
        # lambda=0, psi=0 -> Identity
        # lambda=0, psi=1 -> Global Context / mean across neurons
        y_mean = y.mean(dim=1, keepdim=True)
        contextual_branch = (1.0 - psi) * y + psi * y_mean

        # Exact paper equation.
        return lam * probabilistic_branch + (1.0 - lam) * contextual_branch


# -----------------------------------
# Traditional Neuron + Lambda-Psi Field
# -----------------------------------
class traditional_neuron_lambda_psi_field_layer(nn.Module):
    r"""Traditional affine neuron layer followed by the Lambda-Psi field.

    Stage 1:
        y_j = sum_i w_ji x_i + b_j

    Stage 2:
        m_j = NeuronField(y_j; lambda, psi)

    No ReLU is applied internally.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        lambda_init: float = 0.5,
        psi_init: float = 0.5,
        trainable_field: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.neuron_layer = nn.Linear(
            in_features=self.input_dim,
            out_features=self.output_dim,
            bias=use_bias,
        )
        self.field = lambda_psi_field_layer(
            lambda_init=lambda_init,
            psi_init=psi_init,
            trainable_field=trainable_field,
            eps=eps,
        )

    @property
    def lambda_value(self) -> torch.Tensor:
        return self.field.lambda_value

    @property
    def psi_value(self) -> torch.Tensor:
        return self.field.psi_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.neuron_layer(x)
        return self.field(y)


# -----------------------------------
# APTx Neuron + Lambda-Psi Field
# -----------------------------------
class aptx_neuron_lambda_psi_field_layer(nn.Module):
    r"""APTx Neuron layer followed by the Lambda-Psi field.

    Stage 1 uses the published APTx Neuron formulation.  Stage 2 applies the same
    neuron-agnostic Lambda-Psi field used by ``traditional_neuron_lambda_psi_field_layer``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_alpha_trainable: bool = True,
        use_delta: bool = True,
        lambda_init: float = 0.5,
        psi_init: float = 0.5,
        trainable_field: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.aptx_neuron_layer = aptx_neuron_layer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            is_alpha_trainable=is_alpha_trainable,
            use_delta=use_delta,
        )
        self.field = lambda_psi_field_layer(
            lambda_init=lambda_init,
            psi_init=psi_init,
            trainable_field=trainable_field,
            eps=eps,
        )

    @property
    def lambda_value(self) -> torch.Tensor:
        return self.field.lambda_value

    @property
    def psi_value(self) -> torch.Tensor:
        return self.field.psi_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.aptx_neuron_layer(x)
        return self.field(y)
