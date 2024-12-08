from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Implementation for Task 4.3.
    # Calculate the new height and width after pooling using integer division
    new_height = height // kh
    new_width = width // kw
    # Reshape to split both height and width dimensions into their kernel components
    # From: (batch, channel, height, width)
    # To:   (batch, channel, new_height, kh, new_width, kw)
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # Rearrange dimensions to get pooling windows together
    # From: (batch, channel, new_height, kh, new_width, kw)
    # To:   (batch, channel, new_height, new_width, kh * kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D over a tensor using a given kernel size

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width that has been averaged over the pooling kernel

    """
    # Implementation for Task 4.3.
    # Get the shape of the input tensor
    batch, channel, _, _ = input.shape
    # Reshape the input tensor using the tile function
    tiled, new_height, new_width = tile(input, kernel)
    # Average over the pooling window and ensure correct output shape
    return tiled.mean(dim=-1).view(batch, channel, new_height, new_width)


# Implementation for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width x 1 that is 1 in the location of the max value and 0 otherwise

    """
    # Apply the max operation using the max_reduce function
    out = max_reduce(input, dim)
    # Create a 1-hot tensor with 1s in the positions of the max values and 0s otherwise
    return out == input


class Max(Function):
    """Max operator for the maxpool2d function"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the max operation

        Args:
        ----
            ctx: Context object for storing values for backward pass
            input: batch x channel x height x width
            dim: dimension to reduce over

        Returns:
        -------
            Tensor of size batch x channel x height x width that is 1 in the location of the max value and 0 otherwise

        """
        # Convert the dimension to an integer
        dim_int = int(dim.item())
        # Save the input and dimension for backward pass
        ctx.save_for_backward(input, dim_int)
        # Apply the max reduction
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the max operation

        Args:
        ----
            ctx: Context object for storing values for backward pass
            grad_output: gradient of the output tensor

        Returns:
        -------
            Gradient of the input tensor

        """
        # Retrieve saved values
        input, dim = ctx.saved_values
        # Compute gradient using argmax
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction over a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width with max reduction applied to the specified dimension

    """
    # Apply the max operation using the Max class, and ensure the dimension is a tensor
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply softmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with softmax applied to the specified dimension

    """
    # Compute the exp of the input tensor
    exp_input = input.exp()
    # Compute the sum of the exp values along the specified dimension
    sum_exp = exp_input.sum(dim=dim)
    # Compute the softmax by dividing the exp values by the sum of the exp values
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply logsoftmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with logsoftmax applied to the specified dimension

    """
    # Compute the log of the softmax by subtracting the log of the sum of the exp values from the input tensor
    return input - input.exp().sum(dim=dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling 2D over a tensor using a given kernel size

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width with max pooling applied

    """
    # Get the shape of the input tensor
    batch, channel, height, width = input.shape
    # Reshape the input tensor using the tile function
    tiled, new_height, new_width = tile(input, kernel)
    # Apply max reduction over the pooling window
    return max(tiled, dim=-1).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor

    Args:
    ----
        input: batch x channel x height x width
        rate: dropout rate
        ignore: if True, return the input tensor unchanged

    Returns:
    -------
        Tensor of size batch x channel x height x width with dropout applied

    """
    # If ignore is True or the rate is 0, return the input tensor
    if ignore or rate == 0:
        return input
    else:
        # Generate random noise tensor with the same shape as the input tensor
        mask = rand(input.shape) > rate
        # Apply the mask to the input tensor
        return input * mask
