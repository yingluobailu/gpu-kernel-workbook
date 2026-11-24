// Comparison for exercise 02. Each lane reads a column and writes a strided result.
extern "C" __global__ void solve(const float* src, float* dst, int rows, int cols, int pitch) {
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int first = blockIdx.y * 32 + threadIdx.y;
    for (int r = first; r < first + 32 && r < rows; r += blockDim.y) {
        if (col < cols) dst[col * rows + r] = src[r * pitch + col];
    }
}
