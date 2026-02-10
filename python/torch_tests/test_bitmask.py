# mypy: ignore-errors

import torch
import numpy as np
import pytest
from numpy.typing import NDArray

from typing import List, Tuple

from llguidance.numpy import get_bitmask_shape
import llguidance.numpy
import llguidance.torch

try:
    import llguidance.mlx as ll_mlx
except ImportError:
    ll_mlx = None # type: ignore

has_cuda = torch.cuda.is_available()
dev = "cuda" if has_cuda else "cpu"


def u32_to_bool(arr: NDArray[np.uint32]) -> NDArray[np.uint8]:
    arr = arr.astype(np.uint32)
    bool_arr = np.unpackbits(arr.view(np.uint8), bitorder="little").reshape(
        arr.shape[0], arr.shape[1] * 32
    )
    return bool_arr


def measure_gpu_time(func, *args, **kwargs): # type: ignore
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()  # Ensure previous ops are finished
    start.record()

    func(*args, **kwargs)  # Run the function

    end.record()
    torch.cuda.synchronize()  # Wait for all ops to finish

    return start.elapsed_time(end)  # Returns time in milliseconds


def gen_test_data(
    batch: int, vocab: int, add_vocab: int
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.random.rand(batch, vocab + add_vocab).astype(np.float32)
    mask = np.random.randint(0, 2**32, get_bitmask_shape(batch, vocab), dtype=np.uint32)
    mask = mask.astype(np.int32)
    return data, mask


def validate_mask_data(data_np: np.ndarray, mask_np: np.ndarray, md_np: np.ndarray):
    cutoff = min(32 * mask_np.shape[1], data_np.shape[1])
    bool_mask = u32_to_bool(mask_np)[:, :cutoff]
    md_cut = md_np[:, :cutoff]
    data_cut = data_np[:, :cutoff]
    assert np.all(md_cut[bool_mask == 0] == -np.inf)
    assert np.all(md_cut[bool_mask == 1] == data_cut[bool_mask == 1])
    assert np.all(md_np[:, cutoff:] == -np.inf)


def gen_test_datas():
    yield gen_test_data(1, 32000, 0)
    yield gen_test_data(3, 32000, 0)
    yield gen_test_data(10, 32000, 0)
    yield gen_test_data(3, 128000, 0)
    for v in range(32001, 32033):
        yield gen_test_data(3, v, 0)
        yield gen_test_data(3, v, 137)
    for v in range(1, 33):
        yield gen_test_data(3, 32007, v)


def run_test(cb):
    for data_np, mask_np in gen_test_datas():
        res = cb(data_np, mask_np)
        validate_mask_data(data_np, mask_np, res)


def test_mask_data_torch():
    def torch_test(data_np: np.ndarray, mask_np: np.ndarray):
        data = torch.tensor(data_np, device=dev)
        mask = torch.tensor(mask_np, device=dev)

        masked_data = data.clone()
        llguidance.torch.apply_token_bitmask_inplace(masked_data, mask)

        return masked_data.cpu().numpy()

    run_test(torch_test)


def test_mask_data_numpy():
    def numpy_test(data_np: np.ndarray, mask_np: np.ndarray):
        masked_data = data_np.copy()
        llguidance.numpy.apply_token_bitmask_inplace(masked_data, mask_np)
        return masked_data

    run_test(numpy_test)


def test_mask_data_mlx():
    if not ll_mlx:
        pytest.skip("mlx is not available")

    import mlx.core as mx

    def mlx_test(data_np: np.ndarray, mask_np: np.ndarray):
        assert ll_mlx
        data = mx.array(data_np)
        masked_data = ll_mlx.apply_token_bitmask(data, mask_np)
        return np.array(masked_data)

    run_test(mlx_test)


def test_mask_data_perf():
    if not has_cuda:
        pytest.skip("CUDA is not available")
    times = []
    for batch in [100, 105, 95, 107, 108, 99]:
        data_np, mask_np = gen_test_data(batch, 128000, 0)
        data = torch.tensor(data_np, device=dev)
        mask = torch.tensor(mask_np, device=dev)
        for _ in range(10):
            masked_data = data.clone()
            t = measure_gpu_time(
                llguidance.torch.apply_token_bitmask_inplace, masked_data, mask
            )
            times.append(round(t * 1000))
    print(times)


if __name__ == "__main__":
    test_mask_data_perf()
