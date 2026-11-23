struct Candidate {
    float value;
    int column;
};

struct Winners {
    Candidate first;
    Candidate second;
};

__device__ bool before(Candidate a, Candidate b) {
    return a.column >= 0 && (b.column < 0 || a.value > b.value || (a.value == b.value && a.column < b.column));
}

__device__ Winners insert(Winners best, Candidate item) {
    if (before(item, best.first)) {
        best.second = best.first;
        best.first = item;
    } else if (item.column != best.first.column && before(item, best.second)) {
        best.second = item;
    }
    return best;
}

__device__ Candidate from_lane(Candidate value, int distance) {
    return {__shfl_down_sync(0xffffffffu, value.value, distance),
            __shfl_down_sync(0xffffffffu, value.column, distance)};
}

__device__ Winners warp_winners(Winners best, int lane) {
    for (int stride = 16; stride > 0; stride /= 2) {
        const Candidate a = from_lane(best.first, stride);
        const Candidate b = from_lane(best.second, stride);
        if (lane + stride < 32) {
            best = insert(insert(best, a), b);
        }
    }
    return best;
}

extern "C" __global__ void solve(const float* scores, const int* keep, float* values, int* indices,
                                 int rows, int cols) {
    __shared__ Winners winners[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const Candidate missing = {-__int_as_float(0x7f800000), -1};
    const Winners empty = {missing, missing};
    Winners best = empty;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int index = row * cols + col;
        if (keep[index]) best = insert(best, {scores[index], col});
    }
    best = warp_winners(best, lane);
    if (lane == 0) winners[warp] = best;
    __syncthreads();

    if (warp == 0) {
        best = warp_winners(lane < 8 ? winners[lane] : empty, lane);
        if (lane == 0) {
            values[2 * row] = best.first.value;
            values[2 * row + 1] = best.second.value;
            indices[2 * row] = best.first.column;
            indices[2 * row + 1] = best.second.column;
        }
    }
}
