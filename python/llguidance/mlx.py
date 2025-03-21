# mypy: warn_unused_ignores = False

from typing import Tuple, List, cast
import numpy as np
import mlx.core as mx # type: ignore[import-not-found]
from ._lib import LLMatcher
from .numpy import get_bitmask_shape, allocate_token_bitmask, fill_next_token_bitmask
from numpy.typing import NDArray


@mx.custom_function # type: ignore[misc]
def apply_token_bitmask_kernel(data: mx.array, mask: mx.array) -> mx.array:
    source = """
        uint batch = thread_position_in_grid.y;  // Batch index
        uint elem = thread_position_in_grid.x;   // Element index within batch

        // Bounds check to prevent out-of-bounds access
        // assert(batch < inp_shape[0] && elem < inp_shape[1]);

        uint word_idx = elem / 32;  // Which u32 word
        uint bit_idx = elem % 32;   // Which bit in the word

        // Bounds check for mask access
        // assert(word_idx < mask_shape[1] && batch < mask_shape[0]);

        uint bit = word_idx < mask_shape[1] && (mask[batch * mask_shape[1] + word_idx] >> bit_idx) & 1;
        out[batch * inp_shape[1] + elem] = bit ? inp[batch * inp_shape[1] + elem] : neg_inf[0];
    """

    kernel = mx.fast.metal_kernel(
        name="bitmask_apply_batched",
        input_names=["inp", "mask", "neg_inf"],
        output_names=["out"],
        source=source,
    )

    # Create neg_inf as a tensor
    neg_inf = mx.array([-float("inf")], dtype=data.dtype)

    outputs = kernel(
        inputs=[data, mask, neg_inf],
        template=[("T", data.dtype)],  # Generic dtype support
        grid=(data.shape[1], data.shape[0],
              1),  # Process all elements across batches
        threadgroup=(256, 1, 1),  # Optimize workgroups
        output_shapes=[data.shape],
        output_dtypes=[data.dtype],
    ) # type: ignore[operator]

    a: mx.array = outputs[0]
    return a


def apply_token_bitmask(logits: mx.array,
                        mask_np: NDArray[np.int32]) -> mx.array:
    mask = mx.array(mask_np)
    if len(logits.shape) == 1:
        logits = mx.expand_dims(logits, axis=0)
    if len(mask.shape) == 1:
        mask = mx.expand_dims(mask, axis=0)
    assert mask.dtype == mx.int32, "Mask must be int32"
    assert len(logits.shape) == 2, "Logits must be 2D"
    batch, vocab = logits.shape
    m_batch, m_vocab = mask.shape
    assert batch == m_batch, "Batch size mismatch"
    r = cast(mx.array, apply_token_bitmask_kernel(logits, mask))
    return r
