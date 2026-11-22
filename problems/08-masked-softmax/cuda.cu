template <bool maximum>
__device__ float combine(float a, float b) {
    return maximum ? fmaxf(a, b) : a + b;
}

template <bool maximum>
__device__ float warp_fold(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = combine<maximum>(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <bool maximum>
__device__ float block_fold(float value, float neutral, float* partials) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    value = warp_fold<maximum>(value);
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = warp_fold<maximum>(lane < 8 ? partials[lane] : neutral);
        if (lane == 0) partials[0] = value;
    }
    __syncthreads();
    const float result = partials[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void solve(const float* src, const int* lengths, float* dst, int rows, int cols) {
    __shared__ float partials[8];
    const int row = blockIdx.x;
    const int valid = lengths[row];
    if (valid == 0) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) dst[row * cols + i] = 0.0f;
        return;
    }
    const float lowest = -__int_as_float(0x7f800000);
    float local_max = lowest;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) local_max = fmaxf(local_max, src[row * cols + i]);
    const float largest = block_fold<true>(local_max, lowest, partials);
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) local_sum += expf(src[row * cols + i] - largest);
    const float sum = block_fold<false>(local_sum, 0.0f, partials);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dst[row * cols + i] = i < valid ? expf(src[row * cols + i] - largest) / sum : 0.0f;
    }
}
