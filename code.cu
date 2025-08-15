
//////////////////////////////////////////////////////////////
// CUDA MATRIX FACTORIZATION WITH TIMING OPTIMIZATIONS
//
// PERFORMANCE OPTIMIZATIONS:
// 1. Pre-initialized random states to avoid expensive curand_init() 
//    calls during timed sections (was causing 400ms+ delays)
// 2. GPU warmup kernels to ensure consistent timing measurements
// 3. Proper synchronization to avoid cold start penalties
//////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

// CUDA error checking utility function
inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

//////////////////////////////////////////////////////////////
// CUDA KERNELS FOR MATRIX FACTORIZATION
//////////////////////////////////////////////////////////////

/**
 * Model 1.0: Basic implementation with no optimizations
 * Uses shared memory for dot product computation
 * P: row-wise storage, Q: row-wise storage
 */
__global__ void model_1(float alpha, float lambda, int f, int iterations, curandState *state, 
                       float *p, float *q, float *R_r, int *R_u, int *R_v, 
                       int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    __shared__ float tmp_products_array[32];
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            float r = R_r[offset];
            int u = R_u[offset];
            int v = R_v[offset];
            
            float tmp_p1 = p[u*K + thread];
            float tmp_q1 = q[N * thread + v];
            
            // Compute dot product using shared memory
            tmp_products_array[thread] = tmp_p1 * tmp_q1;
            __syncthreads();
            
            float tmp_product = 0;
            for (int thread_i = 0; thread_i < 32; thread_i++) {
                tmp_product += tmp_products_array[thread_i];
            }
            float ruv = r - tmp_product;
            
            // Update P and Q matrices
            // Note: Only works for K = blockDim.x = 32
            p[u*K + thread + 0] = tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1);
            q[N * thread + v + 0] = tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1);
        }
    }
}

/**
 * Model 2.1: Memory coalescing optimization
 * Improves memory access patterns for better performance
 * P: row-wise storage, Q: column-wise storage
 */
__global__ void model_2_1(float alpha, float lambda, int f, int iterations, curandState *state, 
                         float *p, float *q, float *R_r, int *R_u, int *R_v, 
                         int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    __shared__ float tmp_products_array[32];
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            float r = R_r[offset];
            int u = R_u[offset];
            int v = R_v[offset];
            
            // Pre-compute base indices for better memory coalescing
            int base_p = u*K;
            // For column-wise Q storage: Q[factor][item] = Q[factor + item*K]
            int base_q = v*K;
            
            float tmp_p1 = p[base_p + thread];
            // Correct column-wise access: thread + v*K
            float tmp_q1 = q[thread + base_q];
            
            // Compute dot product using shared memory
            tmp_products_array[thread] = tmp_p1 * tmp_q1;
            __syncthreads();
            
            float tmp_product = 0;
            for (int thread_i = 0; thread_i < 32; thread_i++) {
                tmp_product += tmp_products_array[thread_i];
            }
            
            float ruv = r - tmp_product;
            
            // Update P and Q matrices
            // Note: Only works for K = blockDim.x = 32
            p[base_p + thread + 0] = tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1);
            q[thread + base_q + 0] = tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1);
        }
    }
}

/**
 * Model 2.2: Warp shuffling optimization
 * Uses warp shuffle operations for faster dot product computation
 * P: row-wise storage, Q: row-wise storage
 */
__global__ void model_2_2(float alpha, float lambda, int f, int iterations, curandState *state, 
                         float *p, float *q, float *R_r, int *R_u, int *R_v, 
                         int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            float r = R_r[offset];
            int u = R_u[offset];
            int v = R_v[offset];
            
            int base_p = u*K;
            int base_q = v;
            
            float tmp_p1 = p[u*K + thread];
            float tmp_q1 = q[N * thread + v];
            
            // Compute dot product using warp shuffle operations
            float tmp_product = tmp_p1 * tmp_q1;
            tmp_product += __shfl_down(tmp_product, 16);
            tmp_product += __shfl_down(tmp_product, 8);
            tmp_product += __shfl_down(tmp_product, 4);
            tmp_product += __shfl_down(tmp_product, 2);
            tmp_product += __shfl_down(tmp_product, 1);
            
            // Broadcast result to all threads in warp
            tmp_product = __shfl(tmp_product, 0);
            
            float ruv = r - tmp_product;
            
            // Update P and Q matrices
            // Note: Only works for K = blockDim.x = 32
            p[u*K + thread + 0] = tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1);
            q[N * thread + v + 0] = tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1);
        }
    }
}

/**
 * Model 2.3: Half precision optimization
 * Uses 16-bit floating point for reduced memory usage
 * P: row-wise storage, Q: row-wise storage
 */
__global__ void model_2_3(float alpha, float lambda, int f, int iterations, curandState *state, 
                         half *p, half *q, float *R_r, int *R_u, int *R_v, 
                         int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    __shared__ float tmp_products_array[32];
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            float r = R_r[offset];
            int u = R_u[offset];
            int v = R_v[offset];
            
            // Convert half precision to float for computation
            float tmp_p1 = __half2float(p[u*K + thread]);
            float tmp_q1 = __half2float(q[N * thread + v]);
            
            // Compute dot product using shared memory
            tmp_products_array[thread] = tmp_p1 * tmp_q1;
            __syncthreads();
            
            float tmp_product = 0;
            for (int thread_i = 0; thread_i < 32; thread_i++) {
                tmp_product += tmp_products_array[thread_i];
            }
            
            float ruv = r - tmp_product;
            
            // Update P and Q matrices, converting back to half precision
            // Note: Only works for K = blockDim.x = 32
            p[u*K + thread + 0] = __float2half(tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1));
            q[N * thread + v + 0] = __float2half(tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1));
        }
    }
}

/**
 * Model 2.4: Cache optimization
 * Uses read-only cache for rating data access
 * P: row-wise storage, Q: row-wise storage
 */
__global__ void model_2_4(float alpha, float lambda, int f, int iterations, curandState *state, 
                         float *p, float *q, float *R_r, int *R_u, int *R_v, 
                         int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    __shared__ float tmp_products_array[32];
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            
            // Use read-only cache for rating data (__ldg)
            float r = __ldg(&R_r[offset]);
            int u = __ldg(&R_u[offset]);
            int v = __ldg(&R_v[offset]);
            
            float tmp_p1 = p[u*K + thread];
            float tmp_q1 = q[N * thread + v];
            
            // Compute dot product using shared memory
            tmp_products_array[thread] = tmp_p1 * tmp_q1;
            __syncthreads();
            
            float tmp_product = 0;
            for (int thread_i = 0; thread_i < 32; thread_i++) {
                tmp_product += tmp_products_array[thread_i];
            }
            
            float ruv = r - tmp_product;
            
            // Update P and Q matrices
            // Note: Only works for K = blockDim.x = 32
            p[u*K + thread + 0] = tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1);
            q[N * thread + v + 0] = tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1);
        }
    }
}

/**
 * Model 3: All optimizations combined
 * Combines memory coalescing, warp shuffling, half precision, and cache optimizations
 * P: row-wise storage, Q: column-wise storage
 */
__global__ void model_3(float alpha, float lambda, int f, int iterations, curandState *state, 
                       half *p, half *q, float *R_r, int *R_u, int *R_v, 
                       int N, int M, int K, int nnz) {
    
    int block = blockIdx.x;
    int thread = threadIdx.x % 32;
    
    // Random state is now pre-initialized
    
    int start;
    for (int iter = 0; iter < iterations; iter++) {
        start = curand(&state[block]) % nnz;
        
        // Process f consecutive samples
        for (int i = 0; i < f; i++) {
            int offset = (start + i) % nnz;
            
            // Use read-only cache for rating data
            float r = __ldg(&R_r[offset]);
            int u = __ldg(&R_u[offset]);
            int v = __ldg(&R_v[offset]);
            
            // Pre-compute base indices for memory coalescing
            int base_p = u*K;
            int base_q = v*K;
            
            // Convert half precision to float for computation
            float tmp_p1 = __half2float(p[base_p + thread]);
            // Correct column-wise access for Q: thread + v*K
            float tmp_q1 = __half2float(q[thread + base_q]);
            
            // Compute dot product using warp shuffle operations
            // Note: Only works for K = blockDim.x = 32
            float tmp_product = tmp_p1 * tmp_q1;
            tmp_product += __shfl_down(tmp_product, 16);
            tmp_product += __shfl_down(tmp_product, 8);
            tmp_product += __shfl_down(tmp_product, 4);
            tmp_product += __shfl_down(tmp_product, 2);
            tmp_product += __shfl_down(tmp_product, 1);
            
            // Broadcast result to all threads in warp
            tmp_product = __shfl(tmp_product, 0);
            float ruv = r - tmp_product;
            
            // Update P and Q matrices, converting back to half precision
            // Note: Only works for K = blockDim.x = 32
            p[base_p + thread + 0] = __float2half(tmp_p1 + alpha*(ruv*tmp_q1 - lambda*tmp_p1));
            q[thread + base_q + 0] = __float2half(tmp_q1 + alpha*(ruv*tmp_p1 - lambda*tmp_q1));
        }
    }
}

// Add this new kernel after the existing kernels, before the utility functions
__global__ void init_random_states(curandState *state, int num_blocks) {
    int block = blockIdx.x;
    if (block < num_blocks) {
        curand_init(clock() + block, 0, block, &state[block]);
    }
}

// Add this warmup kernel after the init_random_states kernel
__global__ void warmup_kernel() {
    // Simple computation to warm up the GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    volatile float dummy = sinf((float)idx) * cosf((float)idx);
}

//////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS AND DATA STRUCTURES
//////////////////////////////////////////////////////////////

/**
 * Data structure to hold training data for matrix factorization
 * r: array of rating values
 * u: array of user indices
 * v: array of item indices
 */
struct RTrain {
    float* r;
    int* u;
    int* v;
};

/**
 * Initialize an array with random values between 0 and 1
 * @param array: pointer to array to initialize
 * @param nrows: number of rows
 * @param ncols: number of columns
 */
void randArray(float* array, int nrows, int ncols) {
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            array[row * ncols + col] = rand() % 100 / 100.0;
        }
    }
}

/**
 * Display an array in matrix format
 * @param array: pointer to array to display
 * @param nrows: number of rows
 * @param ncols: number of columns
 * @param mode: 'l' for row-wise storage, 'c' for column-wise storage
 */
void printArray(float* array, int nrows, int ncols, char mode) {
    // If the matrix is stored in row-wise format
    if (mode == 'l') {
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                std::cout << array[row * ncols + col] << "\t";
            }
            std::cout << "\n";
        }
    }
    // If the matrix is stored in column-wise format
    else if (mode == 'c') {
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                std::cout << array[col * nrows + row] << "\t";
            }
            std::cout << "\n";
        }
    }
}

/**
 * Matrix multiplication: R = P * Q
 * P is stored row-wise, Q is stored column-wise, output R is row-wise
 * @param p: pointer to matrix P
 * @param q: pointer to matrix Q
 * @param M: number of rows in P
 * @param N: number of columns in Q
 * @param K: number of columns in P (= number of rows in Q)
 * @return: pointer to result matrix R
 */
float* prodMat(float* p, float* q, int M, int N, int K) {
    float *r = new float[M * N];
    float sumProd = 0;
    
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            sumProd = 0;
            for (int k = 0; k < K; k++) {
                sumProd = sumProd + p[row * K + k] * q[k + col * K];
            }
            r[row * N + col] = sumProd;
        }
    }
    return r;
}

/**
 * Matrix multiplication: R = P * Q
 * Both P and Q are stored row-wise, output R is row-wise
 * @param p: pointer to matrix P
 * @param q: pointer to matrix Q
 * @param M: number of rows in P
 * @param N: number of columns in Q
 * @param K: number of columns in P (= number of rows in Q)
 * @return: pointer to result matrix R
 */
float* prodMatRowRow(float* p, float* q, int M, int N, int K) {
    float *r = new float[M * N];
    float sumProd = 0;
    
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            sumProd = 0;
            for (int k = 0; k < K; k++) {
                sumProd = sumProd + p[row * K + k] * q[N * k + col];
            }
            r[row * N + col] = sumProd;
        }
    }
    return r;
}

/**
 * Transpose a matrix from row-wise to column-wise storage (or vice versa)
 * @param array: pointer to input matrix
 * @param nrows: number of rows
 * @param ncols: number of columns
 * @return: pointer to transposed matrix
 */
float* transpose(float* array, int nrows, int ncols) {
    float* result = new float[nrows * ncols];
    int result_row, result_col, result_nrows, result_ncols;
    result_nrows = ncols; 
    result_ncols = nrows;
    
    // Row and col correspond to the rows and columns of the input array
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            result_row = col;
            result_col = row;
            result[result_row * result_ncols + result_col] = array[row * ncols + col];
        }
    }
    return result;
}

/**
 * Apply permutation to matrix elements
 * @param array: pointer to input matrix
 * @param nrows: number of rows
 * @param ncols: number of columns
 * @param permutation_row: row permutation array
 * @param permutation_col: column permutation array
 * @param matrix_representation: 'l' for row-wise, 'c' for column-wise
 * @return: pointer to permuted matrix
 */
float* permutation(float* array, int nrows, int ncols, int* permutation_row, int*permutation_col, char matrix_representation) {
    float* array_per = new float[nrows * ncols];
    
    // If matrix is stored row-wise
    if (matrix_representation == 'l') {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                array_per[row * ncols + col] = array[permutation_row[row] * ncols + permutation_col[col]];
            }
        }
    }
    // If matrix is stored column-wise
    else if (matrix_representation == 'c') {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                array_per[col * nrows + row] = array[permutation_col[col] * nrows + permutation_row[row]];
            }
        }
    }
    
    return array_per;
}

/**
 * Apply inverse permutation to matrix elements
 * @param array_per: pointer to permuted matrix
 * @param nrows: number of rows
 * @param ncols: number of columns
 * @param permutation_row: row permutation array
 * @param permutation_col: column permutation array
 * @param matrix_representation: 'l' for row-wise, 'c' for column-wise
 * @return: pointer to inverse permuted matrix
 */
float* inversepermutation(float* array_per, int nrows, int ncols, int* permutation_row, int* permutation_col, char matrix_representation) {
    float* array_inv_per = new float[nrows * ncols];
    
    // If matrix is stored row-wise
    if (matrix_representation == 'l') {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                array_inv_per[permutation_row[row] * ncols + permutation_col[col]] = array_per[row * ncols + col];
            }
        }
    }
    // If matrix is stored column-wise
    else if (matrix_representation == 'c') {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                array_inv_per[permutation_col[col] * nrows + permutation_row[row]] = array_per[col * nrows + row];
            }
        }
    }
    return array_inv_per;
}

/**
 * Create an array with consecutive integers from 0 to size-1
 * @param size: size of the array
 * @return: pointer to range array
 */
int* range(int size) {
    int* range_arr = new int[size];
    for (int i = 0; i < size; i++) {
        range_arr[i] = i;
    }
    return range_arr;
}

/**
 * Create an array with consecutive floats from 0 to size-1
 * @param size: size of the array
 * @return: pointer to range array
 */
float* rangeFloat(int size) {
    float* range_arr = new float[size];
    for (int i = 0; i < size; i++) {
        range_arr[i] = (float)i;
    }
    return range_arr;
}

/**
 * Create an array filled with a specific value
 * @param size: size of the array
 * @param value: value to fill the array with
 * @return: pointer to filled array
 */
float* repFloat(int size, float value) {
    float* range_arr = new float[size];
    for (int i = 0; i < size; i++) {
        range_arr[i] = value;
    }
    return range_arr;
}

/**
 * Select nnz random cells from r_init to create training data
 * @param r_init: pointer to initial rating matrix
 * @param M: number of users
 * @param N: number of items
 * @param nnz: number of non-zero entries to select
 * @return: RTrain structure containing training data
 */
RTrain randomRTrain(float* r_init, int M, int N, int nnz) {
    float* rand_tab = new float[M * N];
    struct RTrain r_train;
    r_train.r = new float[nnz];
    r_train.u = new int[nnz];
    r_train.v = new int[nnz];
    int count = 0;
    
    // Create an array with nnz ones and (M*N - nnz) zeros
    for (int i = 0; i < M * N; i++) {
        if (i < nnz) {
            rand_tab[i] = 1;
        }
        else {
            rand_tab[i] = 0;
        }
    }
    
    // Randomly shuffle the array
    std::random_shuffle(&rand_tab[0], &rand_tab[M*N]);
    
    // Build the training data arrays
    count = 0;
    for (int i = 0; i < M * N; i++) {
        if (rand_tab[i] == 1) {
            r_train.r[count] = r_init[i];
            r_train.u[count] = (int)i / N;  // User index
            r_train.v[count] = (int)i % N;  // Item index
            count++;
        }
    }
    return r_train;
}

/**
 * CUDA kernel to convert float matrices to half precision
 * @param p: input float P matrix
 * @param q: input float Q matrix
 * @param p_half: output half precision P matrix
 * @param q_half: output half precision Q matrix
 * @param M: number of users
 * @param K: number of latent factors
 * @param N: number of items
 */
__global__ void kernel_float2half(float *p, float *q, half *p_half, half *q_half, int M, int K, int N) {
    for (int i = 0; i < M*K; i++) {
        p_half[i] = __float2half(p[i]);
    }
    for (int i = 0; i < K*N; i++) {
        q_half[i] = __float2half(q[i]);
    }
}

/**
 * CUDA kernel to convert half precision matrices to float
 * @param p_half: input half precision P matrix
 * @param q_half: input half precision Q matrix
 * @param p: output float P matrix
 * @param q: output float Q matrix
 * @param M: number of users
 * @param K: number of latent factors
 * @param N: number of items
 */
__global__ void kernel_half2float(half *p_half, half *q_half, float *p, float *q, int M, int K, int N) {
    for (int i = 0; i < M*K; i++) {
        p[i] = __half2float(p_half[i]);
    }
    for (int i = 0; i < K*N; i++) {
        q[i] = __half2float(q_half[i]);
    }
}

/**
 * Compute L2 distance between two matrices
 * @param R_pred: predicted rating matrix
 * @param R: actual rating matrix
 * @param M: number of users
 * @param N: number of items
 * @return: L2 distance
 */
float distanceL2(float* R_pred, float* R, int M, int N) {
    float dist = 0;
    for (int i = 0; i < M*N; i++) {
        dist = dist + (R_pred[i] - R[i])*(R_pred[i] - R[i]);
    }
    return sqrt(dist);
}

//////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////

int main() {
    
    //////////////////////////////////////////////////////////////
    // PROBLEM PARAMETERS
    //////////////////////////////////////////////////////////////
    
    int M = 2048;   // Number of users
    int N = 512;    // Number of items
    int K = 32;     // Number of latent factors
    
    // 80% of cells filled in matrix R
    int nnz = (int)M * N * 0.8;
    
    // Stochastic gradient descent parameters
    float alpha = 0.01;   // Learning rate
    float lambda = 0.001; // Regularization parameter
    
    //////////////////////////////////////////////////////////////
    // MODEL SELECTION FLAGS
    //////////////////////////////////////////////////////////////
    
    bool bool_model_0 = true;    // CPU implementation
    bool bool_model_1 = true;    // GPU basic
    bool bool_model_2_1 = true;  // GPU + memory coalescing
    bool bool_model_2_2 = true;  // GPU + warp shuffling
    bool bool_model_2_3 = true;  // GPU + half precision
    bool bool_model_2_4 = true;  // GPU + cache optimization
    bool bool_model_3 = true;    // GPU + all optimizations
    
    //////////////////////////////////////////////////////////////
    // MATRIX INITIALIZATION
    //////////////////////////////////////////////////////////////
    
    float *h_p, *h_q, *h_r_init;
    h_p = new float[M * K];      // P matrix on host
    h_q = new float[K * N];      // Q matrix on host
    h_r_init = new float[M * N]; // R matrix on host
    struct RTrain h_r_train;     // Training data structure
    
    struct timeval start, end;
    int elapsedtime;
    
    // Initialize random seed
    time_t t;
    srand((unsigned)time(&t));
    
    // Random initialization of P and Q matrices
    randArray(h_p, M, K);
    randArray(h_q, K, N);
    
    // Compute initial R = P * Q
    h_r_init = prodMat(h_p, h_q, M, N, K);
    
    cudaFreeHost(h_p);
    cudaFreeHost(h_q);
    
    //////////////////////////////////////////////////////////////
    // PERMUTATION SETUP FOR RANDOMIZATION
    //////////////////////////////////////////////////////////////
    
    int *permutation_row, *permutation_col, *permutation_identity;
    permutation_identity = range(K);
    permutation_row = range(M);
    permutation_col = range(N);
    
    // Randomly shuffle permutations
    std::random_shuffle(&permutation_row[0], &permutation_row[M]);
    std::random_shuffle(&permutation_col[0], &permutation_col[N]);
    
    // Apply permutation to R matrix
    h_r_init = permutation(h_r_init, M, N, permutation_row, permutation_col, 'l');
    
    // Generate random training matrix
    h_r_train = randomRTrain(h_r_init, M, N, nnz);
    
    //////////////////////////////////////////////////////////////
    // INITIALIZE P AND Q FOR TRAINING
    //////////////////////////////////////////////////////////////
    
    float *p, *q, *q_col;
    p = new float[M * K];      // P matrix for training (always row-wise)
    q = new float[K * N];      // Q matrix for training (row-wise)
    q_col = new float[K * N];  // Q matrix in column-wise format
    randArray(p, M, K);
    randArray(q, K, N);
    
    // Convert Q from row-wise to column-wise for models that need it
    // Row-wise: Q[factor][item] stored as q[factor*N + item]
    // Column-wise: Q[factor][item] stored as q[factor + item*K]
    for (int factor = 0; factor < K; factor++) {
        for (int item = 0; item < N; item++) {
            q_col[factor + item * K] = q[factor * N + item];
        }
    }
    
    // Matrices for inverse permutations (if needed for validation)
    float *inv_perm_p, *inv_perm_q;
    inv_perm_p = new float[M * K];
    inv_perm_q = new float[K * N];
    
    //////////////////////////////////////////////////////////////
    // GPU MEMORY ALLOCATION FOR TRAINING DATA
    //////////////////////////////////////////////////////////////
    
    float *d_R_r;
    int *d_R_u, *d_R_v;
    cudaMalloc((void**)&d_R_r, (size_t)nnz * sizeof(float));
    cudaMalloc((void**)&d_R_u, (size_t)nnz * sizeof(int));
    cudaMalloc((void**)&d_R_v, (size_t)nnz * sizeof(int));
    cudaMemcpy(d_R_r, h_r_train.r, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_u, h_r_train.u, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_v, h_r_train.v, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    //////////////////////////////////////////////////////////////
    // GPU EXECUTION PARAMETERS
    //////////////////////////////////////////////////////////////
    
    int number_of_blocks = 100;    // Number of parallel workers
    int threads_per_block = 32;    // Warp size = 32
    int iterations = 1000;         // Iterations per worker
    int f = 10;                    // Consecutive samples per iteration
    int loop__ = 30;               // Total training loops
    int loop = loop__ - 1;
    
    int numberofgraphs = 10;       // Number of experimental runs
    
    int updates = number_of_blocks * (iterations * f) * loop;
    
    std::cout << "Number of updates = " << updates << "\n\n";
    
    //////////////////////////////////////////////////////////////
    // MODEL 0: CPU IMPLEMENTATION
    //////////////////////////////////////////////////////////////
    
    if (bool_model_0 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            std::cout << "Version 0 (CPU) – Run " << n+1 << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model0/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model0/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Training loop
            for (int m = 0; m < loop; m++) {
                
                gettimeofday(&start, NULL);
                
                // Perform SGD updates
                for (int j = 0; j < number_of_blocks*(iterations*f); j++) {
                    int offset = rand() % nnz;
                    float r = h_r_train.r[offset];
                    int u = h_r_train.u[offset];
                    int v = h_r_train.v[offset];
                    
                    // Compute dot product
                    float tmp_dotproduct = 0;
                    for (int i = 0; i < K; i++) {
                        tmp_dotproduct = tmp_dotproduct + p[u * K + i] * q[N * i + v];
                    }
                    
                    float ruv = r - tmp_dotproduct;
                    
                    // Update P and Q matrices
                    for (int i = 0; i < K; i++) {
                        float tmp_p = p[u*K + i];
                        float tmp_q = q[N*i + v];
                        p[u*K + i] = tmp_p + alpha*(ruv*tmp_q - lambda*tmp_p);
                        q[N*i + v] = tmp_q + alpha*(ruv*tmp_p - lambda*tmp_q);
                    }
                }
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Calculate current distance
                distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 1: GPU BASIC IMPLEMENTATION
    //////////////////////////////////////////////////////////////
    
    if (bool_model_1 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 1 (GPU Basic) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Host matrices
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q;
            
            // Device matrices
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            // Random state for CUDA kernels
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model1/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model1/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&start, NULL);
                
                model_1 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_float, d_q_float, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMatRowRow(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 2.1: GPU WITH MEMORY COALESCING
    //////////////////////////////////////////////////////////////
    
    if (bool_model_2_1 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 2.1 (GPU + Memory Coalescing) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Regenerate column-wise Q for this run
            for (int factor = 0; factor < K; factor++) {
                for (int item = 0; item < N; item++) {
                    q_col[factor + item * K] = q[factor * N + item];
                }
            }
            
            // Host matrices
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q_col;  // Use column-wise Q matrix
            
            // Device matrices
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model21/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model21/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                gettimeofday(&start, NULL);
                
                model_2_1 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_float, d_q_float, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMat(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 2.2: GPU WITH WARP SHUFFLING
    //////////////////////////////////////////////////////////////
    
    if (bool_model_2_2 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 2.2 (GPU + Warp Shuffling) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Host matrices
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q;
            
            // Device matrices
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model22/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model22/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                gettimeofday(&start, NULL);
                
                model_2_2 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_float, d_q_float, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMatRowRow(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 2.3: GPU WITH HALF PRECISION
    //////////////////////////////////////////////////////////////
    
    if (bool_model_2_3 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 2.3 (GPU + Half Precision) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Host matrices (both half and float precision)
            half *h_p_half, *h_q_half;
            h_p_half = new half[M * K];
            h_q_half = new half[K * N];
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q;
            
            // Device matrices (both half and float precision)
            half *d_p_half, *d_q_half;
            size_t size_p_half = M * K * sizeof(half);
            size_t size_q_half = K * N * sizeof(half);
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_half, size_p_half);
            cudaMalloc((void**)&d_q_half, size_q_half);
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            // Convert float to half precision on GPU
            kernel_float2half <<< 1, 1 >>> (d_p_float, d_q_float, d_p_half, d_q_half, M, K, N);
            
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model23/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model23/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                gettimeofday(&start, NULL);
                
                model_2_3 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_half, d_q_half, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Convert half precision back to float for evaluation
                kernel_half2float <<< 1, 1 >>> (d_p_half, d_q_half, d_p_float, d_q_float, M, K, N);
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMatRowRow(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_half);
            cudaFreeHost(h_q_half);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_half);
            cudaFree(d_q_half);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 2.4: GPU WITH CACHE OPTIMIZATION
    //////////////////////////////////////////////////////////////
    
    if (bool_model_2_4 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 2.4 (GPU + Cache Optimization) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Host matrices
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q;
            
            // Device matrices
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model24/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model24/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                gettimeofday(&start, NULL);
                
                model_2_4 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_float, d_q_float, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMatRowRow(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // MODEL 3: GPU WITH ALL OPTIMIZATIONS
    //////////////////////////////////////////////////////////////
    
    if (bool_model_3 == true) {
        
        for (int n = 0; n < numberofgraphs; n++) {
            
            std::cout << "Version 3 (GPU + All Optimizations) – Run " << n+1 << "\n";
            
            randArray(p, M, K);
            randArray(q, K, N);
            
            // Regenerate column-wise Q for this run
            for (int factor = 0; factor < K; factor++) {
                for (int item = 0; item < N; item++) {
                    q_col[factor + item * K] = q[factor * N + item];
                }
            }
            
            // Host matrices (both half and float precision)
            half *h_p_half, *h_q_half;
            h_p_half = new half[M * K];
            h_q_half = new half[K * N];
            float *h_p_float, *h_q_float;
            h_p_float = new float[M * K];
            h_q_float = new float[K * N];
            
            h_p_float = p;
            h_q_float = q_col;  // Use column-wise Q matrix
            
            // Device matrices (both half and float precision)
            half *d_p_half, *d_q_half;
            size_t size_p_half = M * K * sizeof(half);
            size_t size_q_half = K * N * sizeof(half);
            float *d_p_float, *d_q_float;
            size_t size_p_float = M * K * sizeof(float);
            size_t size_q_float = K * N * sizeof(float);
            
            cudaMalloc((void**)&d_p_half, size_p_half);
            cudaMalloc((void**)&d_q_half, size_q_half);
            cudaMalloc((void**)&d_p_float, size_p_float);
            cudaMalloc((void**)&d_q_float, size_q_float);
            cudaMemcpy(d_p_float, h_p_float, size_p_float, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_float, h_q_float, size_q_float, cudaMemcpyHostToDevice);
            
            // Convert float to half precision on GPU
            kernel_float2half <<< 1, 1 >>> (d_p_float, d_q_float, d_p_half, d_q_half, M, K, N);
            
            curandState *rand_state;
            cudaMallocManaged(&rand_state, sizeof(curandState) * number_of_blocks);
            
            // Create output files
            char name[32];
            snprintf(name, sizeof(name), "model3/file_%d.txt", n);
            std::ofstream myfile(name);
            char name1[32];
            snprintf(name1, sizeof(name1), "model3/file_%d(updates).txt", n);
            std::ofstream myfile1(name1);
            
            myfile << "time" << "\t" << "distance" << "\n";
            // myfile1 << "updates per second" << "\n";
            
            float distance;
            float modeltime = 0;
            
            // Initial distance calculation
            distance = distanceL2(prodMatRowRow(p, q, M, N, K), h_r_init, M, N);
            myfile << modeltime << "\t" << distance << "\n";
            std::cout << "Initial distance: " << std::fixed << std::setprecision(0) << distance << "\n";
            
            // Initialize random states once before training
            init_random_states <<< number_of_blocks, 1 >>> (rand_state, number_of_blocks);
            checkCudaErr(cudaDeviceSynchronize(), "Random state initialization");
            
            // Warm up GPU to ensure consistent timing
            warmup_kernel <<< number_of_blocks, threads_per_block >>> ();
            checkCudaErr(cudaDeviceSynchronize(), "GPU warmup");
            
            // Training loop
            for (int i = 0; i < loop; i++) {
                
                gettimeofday(&start, NULL);
                
                model_3 <<< number_of_blocks, threads_per_block >>> (alpha, lambda, f, iterations, rand_state, d_p_half, d_q_half, d_R_r, d_R_u, d_R_v, N, M, K, nnz);
                
                checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
                
                gettimeofday(&end, NULL);
                
                elapsedtime = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
                modeltime = modeltime + elapsedtime;
                
                // Convert half precision back to float for evaluation
                kernel_half2float <<< 1, 1 >>> (d_p_half, d_q_half, d_p_float, d_q_float, M, K, N);
                
                // Copy results back to host
                cudaMemcpy(h_p_float, d_p_float, size_p_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_q_float, d_q_float, size_q_float, cudaMemcpyDeviceToHost);
                
                distance = distanceL2(prodMat(h_p_float, h_q_float, M, N, K), h_r_init, M, N);
                
                // Write to output file
                myfile << modeltime << "\t" << distance << "\n";
            }
            
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << modeltime << " milliseconds" << "\n";
            float updpersec = updates / (modeltime/1000.0);
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << std::left << updpersec << " updates per second" << "\n\n";
            myfile1 << std::fixed << std::setprecision(0) << updpersec << "\n";
            
            myfile.close();
            myfile1.close();
            
            // Cleanup
            cudaFree(rand_state);
            cudaFreeHost(h_p_half);
            cudaFreeHost(h_q_half);
            cudaFreeHost(h_p_float);
            cudaFreeHost(h_q_float);
            cudaFree(d_p_half);
            cudaFree(d_q_half);
            cudaFree(d_p_float);
            cudaFree(d_q_float);
        }
    }
    
    //////////////////////////////////////////////////////////////
    // CLEANUP AND PROGRAM END
    //////////////////////////////////////////////////////////////
    
    // Free host memory
    delete[] p;
    delete[] q;
    delete[] q_col;
    delete[] inv_perm_p;
    delete[] inv_perm_q;
    delete[] h_r_train.r;
    delete[] h_r_train.u;
    delete[] h_r_train.v;
    delete[] permutation_row;
    delete[] permutation_col;
    delete[] permutation_identity;
    
    // Free GPU memory for training data
    cudaFree(d_R_r);
    cudaFree(d_R_u);
    cudaFree(d_R_v);
    
    std::cout << "Program finished successfully.\n";
    
    return 0;
}

