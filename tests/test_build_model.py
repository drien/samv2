import pytest
import torch

from sam2.build_sam import build_sam2
from sam2.utils.download import download_weights
from sam2.utils.misc import variant_to_config_mapping


@pytest.mark.full
@pytest.mark.parametrize(
    "variant",
    [
        "tiny",
        "small",
        "base_plus",
        "large",
        "2.1/tiny",
        "2.1/small",
        "2.1/base_plus",
        "2.1/large",
    ],
)
def test_build_sam(download_weights, variant: str):
    parts = variant.split("/")
    base_variant = parts[-1]
    version = f"{parts[0]}" if len(parts) > 1 else "2"

    model = build_sam2(
        variant_to_config_mapping[variant],
        f"./artifacts/sam{version}_hiera_{base_variant}.pt",
    )

    assert isinstance(model, torch.nn.Module)
