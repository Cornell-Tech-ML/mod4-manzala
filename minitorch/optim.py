from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for all optimizers.

    Args:
    ----
        parameters (Sequence[Parameter]): A sequence of model parameters to optimize.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Args:
    ----
        parameters (Sequence[Parameter]): A sequence of model parameters to optimize.
        lr (float): Learning rate for the optimizer. Defaults to 1.0.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Sets the gradients of all parameters to zero."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Performs a single optimization step, updating parameters based on their gradients."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
