"""Native CUDA queue transition evaluator; no host-side simulation."""
import numpy as np
_CUDA_GLOBAL_SOURCE = r"""
extern "C" __global__
void scan_slide_lanes_global(
    const unsigned long long* area_masks,
    const unsigned char* skip_no_press,
    const int* area_counts,
    const unsigned long long* states,
    const unsigned long long* up_masks,
    const double* times,
    const int* transition_starts,
    const int* transition_ends,
    double* completion_times,
    int* completion_indices,
    const int rows,
    const int max_areas)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    const int area_count = area_counts[row];
    const int begin = transition_starts[row];
    const int end = transition_ends[row];
    if (area_count <= 0 || end <= begin) {
        completion_times[row] = __longlong_as_double(0x7ff8000000000000ULL);
        completion_indices[row] = -1;
        return;
    }
    const int area_base = row * max_areas;
    int index = 0;
    unsigned long long pressing = 0ULL;
    for (int transition = begin; transition < end; ++transition) {
        const unsigned long long state = states[transition];
        const unsigned long long up = up_masks[transition];
        const int iteration_limit = area_count * 3 > 4 ? area_count * 3 : 4;
        for (int iteration = 0; iteration < iteration_limit; ++iteration) {
            bool changed = false;
            if (pressing == 0ULL) {
                const unsigned long long hit = state & area_masks[area_base + index];
                if (hit != 0ULL) {
                    pressing = hit & (~hit + 1ULL);
                    if (index >= area_count - 1) {
                        completion_times[row] = times[transition];
                        completion_indices[row] = transition;
                        return;
                    }
                    changed = true;
                }
            } else if ((state & pressing) == 0ULL) {
                pressing = 0ULL;
                ++index;
                changed = true;
                if (index >= area_count) {
                    completion_times[row] = times[transition];
                    completion_indices[row] = transition;
                    return;
                }
            }
            if (index < area_count - 1) {
                const bool can_skip = pressing != 0ULL || skip_no_press[area_base + index] != 0;
                const unsigned long long hit = (state | up) & area_masks[area_base + index + 1];
                if (can_skip && hit != 0ULL) {
                    pressing = hit & (~hit + 1ULL);
                    ++index;
                    changed = true;
                    if (index >= area_count - 1) {
                        completion_times[row] = times[transition];
                        completion_indices[row] = transition;
                        return;
                    }
                }
            }
            if (!changed) break;
        }
    }
    completion_times[row] = __longlong_as_double(0x7ff8000000000000ULL);
    completion_indices[row] = -1;
}
"""


class CudaSlideQueueScanner:
    block_size=256
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.global_kernel=cp.RawKernel(_CUDA_GLOBAL_SOURCE,'scan_slide_lanes_global',options=('--std=c++17',))

    def scan_global_device(
        self, area_masks, skip_no_press, area_counts, states, up_masks,
        times, transition_starts, transition_ends,
    ):
        """Run every Slide lane against one device-resident chart timeline."""
        cp = self.cp
        area_masks = cp.ascontiguousarray(area_masks, dtype=cp.uint64)
        skip_no_press = cp.ascontiguousarray(skip_no_press, dtype=cp.uint8)
        area_counts = cp.ascontiguousarray(area_counts, dtype=cp.int32)
        states = cp.ascontiguousarray(states, dtype=cp.uint64)
        up_masks = cp.ascontiguousarray(up_masks, dtype=cp.uint64)
        times = cp.ascontiguousarray(times, dtype=cp.float64)
        transition_starts = cp.ascontiguousarray(transition_starts, dtype=cp.int32)
        transition_ends = cp.ascontiguousarray(transition_ends, dtype=cp.int32)
        rows = int(area_masks.shape[0])
        completion = cp.empty(rows, dtype=cp.float64)
        completion_index = cp.empty(rows, dtype=cp.int32)
        blocks = (rows + self.block_size - 1) // self.block_size
        self.global_kernel(
            (blocks,), (self.block_size,),
            (area_masks, skip_no_press, area_counts, states, up_masks, times,
             transition_starts, transition_ends, completion, completion_index,
             np.int32(rows), np.int32(area_masks.shape[1])),
        )
        return completion, completion_index
