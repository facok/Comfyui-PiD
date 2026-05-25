"""Windows compatibility patches for PiD.

PiD uses torch.distributed.checkpoint which is unavailable on Windows.
This module sets up mocks before importing PiD code.
"""

import sys
import types
import warnings


def _patch_torch_distributed_checkpoint():
    """Mock torch.distributed.checkpoint for Windows environments."""
    try:
        import torch.distributed.checkpoint
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import torch.distributed
    except ImportError:
        warnings.warn("torch.distributed not available, skipping checkpoint patch")
        return

    mock_dcp = types.ModuleType("torch.distributed.checkpoint")
    mock_dcp.FileSystemReader = type("FileSystemReader", (), {"__init__": lambda *a, **k: None})
    mock_dcp.FileSystemWriter = type("FileSystemWriter", (), {"__init__": lambda *a, **k: None})
    mock_dcp.load = lambda *a, **k: None
    mock_dcp.save = lambda *a, **k: None
    mock_dcp.state_dict_saver = types.ModuleType("state_dict_saver")
    mock_dcp.state_dict_loader = types.ModuleType("state_dict_loader")
    mock_dcp.default_planner = types.ModuleType("default_planner")
    mock_dcp._storage_utils = types.ModuleType("_storage_utils")
    mock_dcp._storage_utils._storage_setup = lambda *a, **k: None

    torch.distributed.checkpoint = mock_dcp
    sys.modules["torch.distributed.checkpoint"] = mock_dcp
    sys.modules["torch.distributed.checkpoint.state_dict_saver"] = mock_dcp.state_dict_saver
    sys.modules["torch.distributed.checkpoint.state_dict_loader"] = mock_dcp.state_dict_loader
    sys.modules["torch.distributed.checkpoint.default_planner"] = mock_dcp.default_planner
    sys.modules["torch.distributed.checkpoint._storage_utils"] = mock_dcp._storage_utils


def _patch_pid_dcp():
    """Mock PiD's imaginaire DCP checkpointer to avoid import errors."""
    mock_mod = types.ModuleType("pid._ext.imaginaire.checkpointer.dcp")
    mock_mod.DefaultLoadPlanner = type("DefaultLoadPlanner", (), {
        "__init__": lambda self, *a, **k: None,
    })
    mock_mod.DistributedCheckpointer = type("DistributedCheckpointer", (), {
        "__init__": lambda self, *a, **k: None,
        "get_storage_reader": lambda self, *a, **k: None,
    })
    mock_mod.ModelWrapper = type("ModelWrapper", (), {
        "__init__": lambda self, *a, **k: None,
        "state_dict": lambda self: {},
        "load_state_dict": lambda self, *a, **k: None,
    })
    sys.modules["pid._ext.imaginaire.checkpointer.dcp"] = mock_mod


def setup_pid_compat():
    """Call before any PiD imports."""
    _patch_torch_distributed_checkpoint()
    _patch_pid_dcp()
