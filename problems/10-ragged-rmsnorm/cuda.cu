__device__ float warp_sum(float sum) {
    for (int stride = 16; stride; stride /= 2) sum += __shfl_down_sync(0xffffffffu, sum, stride);
    return sum;
}

extern "C" __global__ void solve(const float* x, const float* residual, const float* weights,
                                 const int* lengths, float* y, int rows, int cols) {
    __shared__ float sums[8];
    const int row = blockIdx.x;
    const int start = row * cols;
    const int length = lengths[row];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    float squares = 0.0f;
    for (int i = tid; i < length; i += blockDim.x) {
        const float z = x[start + i] + residual[start + i];
        squares += z * z;
    }
    squares = warp_sum(squares);
    if (lane == 0) sums[warp] = squares;
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) sums[0] = rsqrtf(squares / (length ? length : 1) + 1e-5f);
    }
    __syncthreads();
    const float scale = sums[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        y[start + i] = i < length ? (x[start + i] + residual[start + i]) * scale * weights[i] : 0.0f;
    }
}
