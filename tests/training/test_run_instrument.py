"""Parameter connectivity must distinguish unused heads from zero-valued gradients."""

import pytest
import torch

from equiparity.training.run_instrument import parameter_counts


def test_parameter_counts_exclude_disconnected_heads_but_keep_zero_gradients() -> None:
    model = torch.nn.ModuleDict({"used": torch.nn.Linear(3, 2), "unused": torch.nn.Linear(2, 1)})
    with pytest.raises(ValueError, match="backward"):
        parameter_counts(model)
    # Exact zero loss leaves zero gradients on connected parameters, not None.
    output = model["used"](torch.zeros(4, 3))
    (output - output.detach()).square().sum().backward()
    assert parameter_counts(model) == {
        "allocated": 11,
        "trainable": 11,
        "tensor_loss_connected": 8,
    }
