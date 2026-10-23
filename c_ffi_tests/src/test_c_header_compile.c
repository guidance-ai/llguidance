#include "llguidance.h"

int llguidance_c_header_compiles(void) {
  struct LlgMatcher *matcher = 0;
  struct LlgCancellationHandle *handle = 0;
  struct LlgCancellationHandle *(*get_handle)(struct LlgMatcher *) =
      llg_matcher_get_cancellation_handle;
  struct LlgCancellationHandle *(*clone_handle)(
      const struct LlgCancellationHandle *) = llg_clone_cancellation_handle;
  void (*cancel)(const struct LlgCancellationHandle *) = llg_cancel;
  void (*free_handle)(struct LlgCancellationHandle *) =
      llg_free_cancellation_handle;
  bool (*is_cancelled)(const struct LlgMatcher *) =
      llg_matcher_is_cancelled;

  (void)matcher;
  (void)handle;
  (void)get_handle;
  (void)clone_handle;
  (void)cancel;
  (void)free_handle;
  (void)is_cancelled;
  return 0;
}
