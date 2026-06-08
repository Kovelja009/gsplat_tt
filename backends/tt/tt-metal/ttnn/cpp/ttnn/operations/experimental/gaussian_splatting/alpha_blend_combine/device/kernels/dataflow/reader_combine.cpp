// SPDX-License-Identifier: Apache-2.0
// Combine reader (STUB — Task 6 implements partial reads). Args:
//   0: partials_addr  1: plan_addr  2: plan_start  3: plan_count
#include <cstdint>
#include "dataflow_api.h"
void kernel_main() {
    // TODO(Task 6): for each plan row in [plan_start, plan_start+plan_count),
    // read K jobs' (R,G,B,T) tiles from partials into CB_PARTIAL + push K to CB_META.
}
