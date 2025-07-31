from typing import Tuple, List, Union
import torch
from ._lib import LLMatcher, LLExecutor, LLInterpreter


def get_bitmask_shape(batch_size: int, vocab_size: int) -> Tuple[int, int]:
    return (batch_size, (vocab_size + 31) // 32)


def allocate_token_bitmask(batch_size: int, vocab_size: int) -> torch.Tensor:
    return torch.full(
        get_bitmask_shape(batch_size, vocab_size),
        -1,
        dtype=torch.int32,
        pin_memory=torch.cuda.is_available(),
    )


@torch.compile(dynamic=True)  # faster than dynamic=False and jit.script
def apply_token_bitmask_inplace_kernel(logits: torch.Tensor,
                                       mask: torch.Tensor) -> None:
    mask_expanded = torch.repeat_interleave(mask, 32, dim=1)
    bit_indices = torch.arange(32, device=logits.device,
                               dtype=torch.int32).repeat(mask.shape[1])
    bit_masks = (mask_expanded >> bit_indices) & 1  # Extract each bit
    bit_masks = bit_masks[:, :logits.shape[1]]  # Trim to match vocab size
    logits.masked_fill_(bit_masks == 0, float("-inf"))  # Apply mask


def apply_token_bitmask_inplace(logits: torch.Tensor,
                                mask: torch.Tensor) -> None:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    assert mask.dtype == torch.int32, "Mask must be int32"
    assert logits.dim() == 2, "Logits must be 2D"
    batch, vocab = logits.shape
    m_batch, m_vocab = mask.shape
    assert batch == m_batch, "Batch size mismatch"
    cutoff = 32 * m_vocab
    if vocab > cutoff:
        logits[:, cutoff:] = float("-inf")
        logits = logits[:, :cutoff]
    apply_token_bitmask_inplace_kernel(logits, mask)


def fill_next_token_bitmask(interp: Union[LLMatcher, LLInterpreter],
                            bitmask: torch.Tensor,
                            index: int = 0) -> None:
    assert bitmask.dtype == torch.int32, "Mask must be int32"
    assert bitmask.is_cpu, "Mask must be on CPU"
    assert bitmask.dim() == 2, "Mask must be 2D"
    v = bitmask[index, :]
    assert v.is_contiguous(), "Mask must be contiguous"
    interp.unsafe_compute_mask_ptr(v.data_ptr(), v.numel() * v.element_size())


def fill_next_token_bitmask_par(executor: LLExecutor,
                                matchers: List[Tuple[LLMatcher, int]],
                                bitmask: torch.Tensor) -> None:
    assert bitmask.dtype == torch.int32, "Mask must be int32"
    assert bitmask.is_cpu, "Mask must be on CPU"
    assert bitmask.dim() == 2, "Mask must be 2D"
    batch, vocab = bitmask.shape
    assert bitmask.is_contiguous(), "Mask must be contiguous"
    executor.unsafe_compute_mask_ptr(matchers, bitmask.data_ptr(), vocab * 4,
                                     batch)


def fill_next_token_bitmask_par_with_draft_tokens(executor: LLExecutor,
                                matchers: List[Tuple[LLMatcher, int, List[int]]],
                                bitmask: torch.Tensor) -> None:
    assert bitmask.dtype == torch.int32, "Mask must be int32"
    assert bitmask.is_cpu, "Mask must be on CPU"
    assert bitmask.dim() == 2, "Mask must be 2D"
    batch, vocab = bitmask.shape
    assert bitmask.is_contiguous(), "Mask must be contiguous"
    executor.unsafe_compute_mask_ptr_with_draft_token(matchers, bitmask.data_ptr(), vocab * 4, batch)
