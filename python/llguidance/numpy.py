from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
from ._lib import LLMatcher, LLExecutor


def get_bitmask_shape(batch_size: int, vocab_size: int) -> Tuple[int, int]:
    return (batch_size, (vocab_size + 31) // 32)


def allocate_token_bitmask(batch_size: int,
                           vocab_size: int) -> NDArray[np.int32]:
    return np.full(
        get_bitmask_shape(batch_size, vocab_size),
        -1,
        dtype=np.int32,
    )


def apply_token_bitmask_inplace_kernel(logits: NDArray[np.float32],
                                       mask: NDArray[np.int32]) -> None:
    mask_expanded = np.repeat(mask, 32, axis=1)
    bit_indices = np.tile(np.arange(32, dtype=np.int32), mask.shape[1])
    bit_masks = (mask_expanded >> bit_indices) & 1  # Extract each bit
    bit_masks = bit_masks[:, :logits.shape[1]]  # Trim to match vocab size
    logits[bit_masks == 0] = -np.inf  # Apply mask


def apply_token_bitmask_inplace(logits: NDArray[np.float32],
                                mask: NDArray[np.int32]) -> None:
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)
    if mask.ndim == 1:
        mask = np.expand_dims(mask, axis=0)
    assert mask.dtype == np.int32, "Mask must be int32"
    assert logits.ndim == 2, "Logits must be 2D"
    batch, vocab = logits.shape
    m_batch, m_vocab = mask.shape
    assert batch == m_batch, "Batch size mismatch"
    cutoff = 32 * m_vocab
    if vocab > cutoff:
        logits[:, cutoff:] = -np.inf
        logits = logits[:, :cutoff]
    apply_token_bitmask_inplace_kernel(logits, mask)


def fill_next_token_bitmask(interp: LLMatcher,
                            bitmask: NDArray[np.int32],
                            index: int = 0) -> None:
    assert bitmask.dtype == np.int32, "Mask must be int32"
    assert bitmask.ndim == 2, "Mask must be 2D"
    v = bitmask[index, :]
    assert v.flags["C_CONTIGUOUS"], "Mask must be contiguous"
    interp.unsafe_compute_mask_ptr(v.ctypes.data, v.size * v.itemsize)


def fill_next_token_bitmask_par(executor: LLExecutor,
                                matchers: List[Tuple[LLMatcher, int]],
                                bitmask: NDArray[np.int32]) -> None:
    """
    Compute the token mask directly into the specified array.
    For each matcher, provide the index of the target mask.
    """
    assert bitmask.dtype == np.int32, "Mask must be int32"
    assert bitmask.ndim == 2, "Mask must be 2D"
    batch, vocab = bitmask.shape
    assert bitmask.flags["C_CONTIGUOUS"], "Mask must be contiguous"
    executor.unsafe_compute_mask_ptr(matchers, bitmask.ctypes.data, vocab * 4, batch)


def fill_next_token_bitmask_par_with_draft_tokens(executor: LLExecutor,
                                matchers: List[Tuple[LLMatcher, int, List[int]]],
                                bitmask: NDArray[np.int32]) -> None:
    """
    Compute the token mask directly into the specified array.
    For each matcher, provide the index of the target mask.
    """
    assert bitmask.dtype == np.int32, "Mask must be int32"
    assert bitmask.ndim == 2, "Mask must be 2D"
    batch, vocab = bitmask.shape
    assert bitmask.flags["C_CONTIGUOUS"], "Mask must be contiguous"
    executor.unsafe_compute_mask_ptr_with_draft_token(matchers, bitmask.ctypes.data, vocab * 4, batch)
