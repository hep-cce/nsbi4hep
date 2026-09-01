import pytest
import torch

# Torch-based tests parametrized over this fixture run once per available device:
# always on cpu, and additionally on cuda when the machine has a usable GPU.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)
