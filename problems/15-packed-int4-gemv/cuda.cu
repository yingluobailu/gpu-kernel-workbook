extern "C" __global__ void solve(const unsigned* packed, const float* scale, const int* zero,
                                 const float* x, float* out, int rows, int width, int group_size) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int words = (width + 7) / 8, groups = (width + group_size - 1) / group_size;
    float sum = 0;
    for (int w = tid; w < words; w += blockDim.x) {
        const unsigned word = packed[row * words + w];
        for (int i = 0; i < 8 && w * 8 + i < width; ++i) {
            const int col = w * 8 + i, group = row * groups + col / group_size;
            const int code = (word >> (4 * i)) & 15u;
            sum += (code - zero[group]) * scale[group] * x[col];
        }
    }
    __shared__ float sums[128];
    sums[tid] = sum;
    __syncthreads();
    for (int step = 64; step > 0; step /= 2) {
        if (tid < step) sums[tid] += sums[tid + step];
        __syncthreads();
    }
    if (tid == 0) out[row] = sums[0];
}
