import metalgpu
import numpy as np
import time

# Create the Metal GPU interface
instance = metalgpu.Interface()

# Revised Metal shader for matrix multiplication using a struct for dimensions.
# It expects:
#   buffer(0): device pointer to matrix A (float array of size M*K)
#   buffer(1): device pointer to matrix B (float array of size K*N)
#   buffer(2): device pointer to output matrix C (float array of size M*N)
#   buffer(3): constant buffer with the MatDims struct (holds M, N, and K)
#
# Each thread computes one element of C at (row, col) using the dot product.
shader_str = """
#include <metal_stdlib>
using namespace metal;

struct MatDims {
    uint M;
    uint N;
    uint K;
};

kernel void matmul(device float* A         [[ buffer(0) ]],
                   device const float* B         [[ buffer(1) ]],
                   device float* C               [[ buffer(2) ]],
                   constant MatDims &dims        [[ buffer(3) ]],
                   uint2 gid                   [[ thread_position_in_grid ]])
{
    uint M = dims.M;
    uint N = dims.N;
    uint K = dims.K;

    // Each thread computes one element: row = gid.y, col = gid.x
    if (gid.y >= M || gid.x >= N) return;

    float sum = 0.0;
    for (uint i = 0; i < K; i++) {
        sum += A[gid.y * K + i] * B[i * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum; // Assign the computed sum to C
}
"""

instance.load_shader_from_string(shader_str)
instance.set_function("matmul")

# Define matrix dimensions (change these as needed)
M = 10  # rows in A and C
K = 10  # columns in A, rows in B
N = 10  # columns in B and C

# Create input matrices using numpy (float32 arrays)
A_np = np.random.rand(M, K).astype(np.float32)
B_np = np.random.rand(K, N).astype(np.float32)
# Compute expected result on the CPU for verification
C_expected = np.dot(A_np, B_np)

# Create GPU buffers for matrices (the instance.create_buffer function expects the number of elements)
buffer_A = instance.create_buffer(M * K, "float")
buffer_B = instance.create_buffer(K * N, "float")
buffer_C = instance.create_buffer(M * N, "float")

# Copy the numpy data into the GPU buffers (flattened in row‐major order)
buffer_A.contents[:] = A_np.flatten()
buffer_B.contents[:] = B_np.flatten()

# Pack the dimensions into a single structure.
# Create a numpy array of 3 uint32 values corresponding to [M, N, K].
dims_np = np.array([M, N, K], dtype=np.uint32)
buffer_dims = instance.create_buffer(3, "uint")
buffer_dims.contents[:] = dims_np

# Set up the thread grid.
# Since the output matrix is MxN, we launch a grid of size (N, M, 1)
# where thread id x is the column index and id y is the row index.
thread_size = metalgpu.MetalSize(N, M, 1)

print(buffer_A.contents[0])
# Run the kernel on the GPU and time the execution.
gpu_start = time.time()
instance.run_function(thread_size, [buffer_A, buffer_B, buffer_C, buffer_dims])
gpu_end = time.time()
print("Matrix multiplication GPU time:", gpu_end - gpu_start)

print(buffer_C.contents, buffer_A.contents[0])

np_time = time.time()
mat = np.matmul(A_np, B_np)
np_time = time.time() - np_time

print("Matrix multiplication CPU time:", np_time)

# Retrieve the result from GPU memory and reshape to MxN
C_gpu = np.array(buffer_C.contents[:]).reshape(M, N)

# Verify that the GPU result is close to the CPU result
np.testing.assert_allclose(C_gpu, C_expected, rtol=1e-5)
print("Matrix multiplication result verified!")

# Clean up: release all buffers
buffer_A.release()
buffer_B.release()
buffer_C.release()
buffer_dims.release()

