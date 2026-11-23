__device__ float warp_sum(float sum) {
    for (int stride = 16; stride; stride /= 2) sum += __shfl_down_sync(0xffffffffu, sum, stride);
    return sum;
}

extern "C" __global__ void solve(const int* row_offsets, const int* column_indices, const float* weights,
                                 const float* x, float* y, int rows, int cols) {
    __shared__ float sums[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float sum = 0.0f;
    for (int i = row_offsets[row] + threadIdx.x; i < row_offsets[row + 1]; i += blockDim.x) {
        sum += weights[i] * x[column_indices[i]];
    }
    sum = warp_sum(sum);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) y[row] = sum;
    }
}
