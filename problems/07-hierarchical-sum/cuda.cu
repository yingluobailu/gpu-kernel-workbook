__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void solve(const float* src, float* partials, int n) {
    __shared__ float tile[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int i = blockIdx.x * 512 + threadIdx.x;
    float sum = i < n ? src[i] : 0.0f;
    sum += i + 256 < n ? src[i + 256] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) tile[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < 8 ? tile[lane] : 0.0f);
        if (lane == 0) partials[blockIdx.x] = sum;
    }
}
