"""Utilities for finalizing a module for use within a critic or actor."""

from typing import Tuple

from torch.nn import Linear, Module, Sequential

from decuen.structs import State


def finalize_module(module: Module, in_example: State, out_size: int) -> Tuple[Module, Sequential]:
    """Finalize and verify a given module.

    Verifies that the module is compatible with the given example and constructs a new module that has a one-dimensional
    output of the given size. Returns the final added layer and the newly constructed module.
    """
    try:
        size = module(in_example.tensor).size()
    except RuntimeError:
        raise ValueError("given model is incompatible with the state space")
    if len(size) != 1:
        raise ValueError(f"given model must have one-dimensional output, instead got output shape {size}")

    final_layer = Linear(size[0], out_size)
    return final_layer, Sequential(module, final_layer)
