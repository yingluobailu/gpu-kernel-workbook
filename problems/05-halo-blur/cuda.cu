extern "C" __global__ void solve(const float* src, float* dst, int rows, int cols) {
    __shared__ float tile[18][19];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_y = blockIdx.y * 16;
    const int tile_x = blockIdx.x * 16;
    for (int i = ty * 16 + tx; i < 18 * 18; i += 16 * 16) {
        const int y = tile_y + i / 18 - 1;
        const int x = tile_x + i % 18 - 1;
        tile[i / 18][i % 18] = y >= 0 && y < rows && x >= 0 && x < cols ? src[y * cols + x] : 0.0f;
    }
    __syncthreads();
    if (tile_y + ty < rows && tile_x + tx < cols) {
        const int y = ty + 1;
        const int x = tx + 1;
        const float axial = tile[y-1][x] + tile[y+1][x] + tile[y][x-1] + tile[y][x+1];
        const float corner = tile[y-1][x-1] + tile[y-1][x+1] + tile[y+1][x-1] + tile[y+1][x+1];
        dst[(tile_y + ty) * cols + tile_x + tx] = (4.0f * tile[y][x] + 2.0f * axial + corner) / 16.0f;
    }
}
