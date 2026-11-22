extern "C" __global__ void solve(const float* src, float* dst, int rows, int cols, int pitch) {
    __shared__ float tile[32][33];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int from_col = blockIdx.x * 32 + tx;
    const int from_row = blockIdx.y * 32 + ty;
    for (int step = 0; step < 32; step += 8) {
        if (from_row + step < rows && from_col < cols) {
            tile[ty + step][tx] = src[(from_row + step) * pitch + from_col];
        }
    }
    __syncthreads();

    const int to_col = blockIdx.y * 32 + tx;
    const int to_row = blockIdx.x * 32 + ty;
    for (int step = 0; step < 32; step += 8) {
        if (to_row + step < cols && to_col < rows) {
            dst[(to_row + step) * rows + to_col] = tile[tx][ty + step];
        }
    }
}
