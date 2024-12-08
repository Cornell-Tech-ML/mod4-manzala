import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # Implemented for Task 4.4.
    # Test maximum values along different dimensions of the tensor
    # Dimension 0 (reducing across batch dimension)
    out = minitorch.nn.max(t, 0)  # Find max values along first dimension
    assert_close(
        out[0, 0, 0], max([t[i, 0, 0] for i in range(2)])
    )  # Verify first element is correct max
    assert out.shape == (1, 3, 4)  # Dimension 0 should be reduced to size 1
    # Dimension 1 (reducing across channel dimension)
    out = minitorch.nn.max(t, 1)  # Find max values along second dimension
    assert_close(
        out[0, 0, 0], max([t[0, i, 0] for i in range(3)])
    )  # Verify first element is correct max
    assert out.shape == (2, 1, 4)  # Dimension 1 should be reduced to size 1
    # Dimension 2 (reducing across feature dimension)
    out = minitorch.nn.max(t, 2)  # Find max values along third dimension
    # Verify first element is correct max
    assert_close(out[0, 0, 0], max([t[0, 0, i] for i in range(4)]))
    assert out.shape == (2, 3, 1)  # Dimension 2 should be reduced to size 1
    # Check gradients for each dimension
    for dim in range(3):
        minitorch.grad_check(
            # Function to check: max along dimension dim
            lambda t: minitorch.max(t, dim),
            # Input tensor with small random noise to avoid potential gradient issues at exact values
            t + (minitorch.rand(t.shape) * 1e-4),
        )


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
