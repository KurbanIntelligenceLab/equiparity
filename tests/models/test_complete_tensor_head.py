"""The tensor-product head spans required blocks and preserves the enforced symmetry."""

import pytest
import torch

torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

from equiparity.models.tensor_head import CompleteTensorHead  # noqa: E402


@pytest.mark.parametrize(
    "inputs,outputs,missing",
    [
        ("4x0e+4x1o+4x2e+4x3o", "2x1o+1x2o+1x3o", slice(6, 11)),
        ("4x0e+4x1o+4x2e", "2x0e+2x2e+1x4e", slice(12, 21)),
    ],
)
def test_missing_linear_blocks_have_signal_and_gradients(
    inputs: str, outputs: str, missing: slice
) -> None:
    torch.manual_seed(12)
    head = CompleteTensorHead(inputs, outputs).double()
    features = torch.randn(5, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    prediction = head(features)
    assert prediction[:, missing].abs().max() > 1e-5
    prediction[:, missing].square().sum().backward()
    assert features.grad.abs().sum() > 0
    assert head.product.weight.grad.abs().sum() > 0
    assert bool(torch.all(head.output_mask == 1))
    for determinant in (1, -1):
        rotation = determinant * o3.rand_matrix(dtype=torch.float64)
        din = head.irreps_in.D_from_matrix(rotation)
        dout = head.irreps_out.D_from_matrix(rotation)
        torch.testing.assert_close(
            head(features @ din.T), prediction @ dout.T, atol=2e-6, rtol=2e-6
        )


def test_independent_channel_mixing_can_produce_noncollinear_vectors() -> None:
    torch.manual_seed(2)
    per_channel = o3.Irreps("0e+1o+2e+3o")
    inputs = o3.Irreps([term for _ in range(4) for term in per_channel])
    head = CompleteTensorHead(str(inputs), "2x1o+1x2o+1x3o")
    out = head(torch.randn(5, inputs.dim))
    assert torch.linalg.cross(out[:, :3], out[:, 3:6]).abs().max() > 1e-4


def test_unreachable_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot reach"):
        CompleteTensorHead("4x0e", "1x3o")
