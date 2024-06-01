#include <chrono>
#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

/*
 * +---------------+--------+------------------------+
 * | Input         | Output | #define                |
 * +---------------+--------+------------------------+
 * | __half        | __half | USE_F16_IN_OUT         |
 * | __half        | float  | USE_F16_IN_F32_OUT     |
 * | unsigned char | int    | USE_UINT8_IN_INT32_OUT |
 * | signed char   | int    | USE_INT8_IN_INT32_OUT  |
 * | __nv_bfloat16 | float  | USE_BF16_IN_F32_OUT    |
 * | tf32          | float  | USE_TF32_IN_F32_OUT    |
 * | double        | double | USE_F64_IN_OUT         |
 * | u4            | int    | USE_U4_IN_INT32_OUT    |
 * | s4            | int    | USE_S4_IN_INT32_OUT    |
 * | b1            | int    | USE_B1_IN_INT32_OUT    |
 * +---------------+--------+------------------------+
 * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes
 */
#define USE_F64_IN_OUT

/*
 * +---------------------------+--------------+
 * | Matrix Dimensions (m-n-k) | #define      |
 * +---------------------------+--------------+
 * | 16 x 16 x  16             | MNK_16x16x16 |
 * | 32 x  8 x  16             | MNK_32x8x16  |
 * |  8 x 32 x  16             | MNK_8x32x16  |
 * | 16 x 16 x   8             | MNK_16x16x8  |
 * |  8 x  8 x   4             | MNK_8x8x4    |
 * |  8 x  8 x  32             | MNK_8x8x32   |
 * |  8 x  8 x 128             | MNK_8x8x128  |
 * +---------------------------+--------------+
 * Note that not all dimensions are supported for all element types. See link
 * in previous comment.
 */
#define MNK_8x8x4

#if defined(USE_F64_IN_OUT)
    typedef double INPUT_ELEMENT;
    typedef double OUTPUT_ELEMENT;
    #ifndef MNK_8x8x4
        #error "Selected matrix dimensions are not supported for USE_F64_IN_OUT"
    #endif
#else
    #error "Selected matrix element type is not supported"
#endif

#define VERIFY_TOLERANCE 0.005

const bool          DO_CPU_VERIFY = true;
const bool          DEBUG_OUTPUT  = false;
const INPUT_ELEMENT ALPHA         = 1.234;
const INPUT_ELEMENT BETA          = 5.678;
const unsigned long ROWS_A        = 128;
const unsigned long COLS_A        = 128;
const unsigned long ROWS_B        = COLS_A;
const unsigned long INNER_DIM     = COLS_A;
const unsigned long COLS_B        = 128;
const unsigned long ROWS_C        = ROWS_A;
const unsigned long COLS_C        = COLS_B;
const unsigned long ROWS_OUT      = ROWS_A;
const unsigned long COLS_OUT      = COLS_B;
const bool          TRANSPOSE_A   = false;
const bool          TRANSPOSE_B   = false;
const dim3          NUM_BLOCKS      (16,1,1);
const dim3          NUM_THREADS     (1024,1,1);
// I use the term eFLOPS in this file to refer to "effective FLOPS", which
// means "how fast would you have to do floating point operations with the
// naive gemm algorithm to match the observed speed"
const unsigned long long TOTAL_NAIVE_FLOPS = (2 * INNER_DIM - 1) * (ROWS_OUT * COLS_OUT) + (ROWS_OUT * COLS_OUT) * 3;
const double MAX_FLOPS = 275.0e12;

__global__ void d_gemm(
    INPUT_ELEMENT* A,
    INPUT_ELEMENT* B,
    OUTPUT_ELEMENT* C,
    unsigned long r_A,
    unsigned long inner,
    unsigned long c_B,
    INPUT_ELEMENT alpha,
    INPUT_ELEMENT beta,
    bool t_A,
    bool t_B
) {
    // TODO write fast code here
    // TODO add support for transposition
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row = i / c_B;
    int col = i % c_B;
    C[i] *= beta;
    for (int offset = 0; offset < inner; offset++) {
        C[i] += alpha * A[row * inner + offset] * B[offset * c_B + col];
    }
}

void h_gemm(
    INPUT_ELEMENT* A,
    INPUT_ELEMENT* B,
    OUTPUT_ELEMENT* C,
    unsigned long r_A,
    unsigned long inner,
    unsigned long c_B,
    INPUT_ELEMENT alpha,
    INPUT_ELEMENT beta,
    bool t_A,
    bool t_B
) {
    // CPU gemm to double-check GPU output and compare performance
    // TODO add support for transposition
    for (int row = 0; row < r_A; row++) {
        for (int col = 0; col < c_B; col++) {
            C[row * c_B + col] *= beta;
            for (int offset = 0; offset < inner; offset++) {
                C[row * c_B + col] += alpha * A[row * inner + offset] * B[col + c_B * offset];
            }
        }
    }
}

void initInputMatrix(INPUT_ELEMENT* mat, int len) {
    for (int idx = 0; idx < len; idx++) {
        mat[idx] = static_cast<INPUT_ELEMENT>((rand() % 1000) / 100.0 - 5.0);
    }
}

void initOutputMatrix(OUTPUT_ELEMENT* mat, int len) {
    for (int idx = 0; idx < len; idx++) {
        mat[idx] = static_cast<OUTPUT_ELEMENT>((rand() % 1000) / 100.0 - 5.0);
    }
}

bool verify(OUTPUT_ELEMENT* h_solution, OUTPUT_ELEMENT* h_C) {
    for (int idx = 0; idx < ROWS_C * COLS_C; idx++) {
        if (abs(h_C[idx] - h_solution[idx]) > VERIFY_TOLERANCE) {
            printf(
                "Found output mismatch at (%lu,%lu): device returned %f but solution is %f\n",
                idx / COLS_OUT,
                idx % COLS_OUT,
                h_C[idx],
                h_solution[idx]
            );
            return false;
        }
    }
    return true;
}

void printInputMat(INPUT_ELEMENT* mat, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%f ", mat[r * rows + c]);
        }
        printf("\n");
    }
}

void printOutputMat(OUTPUT_ELEMENT* mat, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%f ", mat[r * rows + c]);
        }
        printf("\n");
    }
}

int main() {
    // Host alloc and init
    size_t SIZE_A = ROWS_A * COLS_A * sizeof(INPUT_ELEMENT);
    printf("A: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_A, COLS_A, sizeof(INPUT_ELEMENT), SIZE_A);
    size_t SIZE_B = ROWS_B * COLS_B * sizeof(INPUT_ELEMENT);
    printf("B: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_B, COLS_B, sizeof(INPUT_ELEMENT), SIZE_B);
    size_t SIZE_C = ROWS_C * COLS_C * sizeof(OUTPUT_ELEMENT);
    printf("C: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_C, COLS_C, sizeof(OUTPUT_ELEMENT), SIZE_C);
    size_t TOTAL_SIZE = SIZE_A + SIZE_B + SIZE_C;
    printf("Total eFLOPS: %llu\n", TOTAL_NAIVE_FLOPS);
    printf("Total bytes: %lu\n", TOTAL_SIZE);
    printf("eFLOPS per byte: %f\n", static_cast<double>(TOTAL_NAIVE_FLOPS) / static_cast<double>(TOTAL_SIZE));
    srand(0);
    INPUT_ELEMENT* h_A = (INPUT_ELEMENT*)malloc(SIZE_A);
    initInputMatrix(h_A, ROWS_A * COLS_A);
    INPUT_ELEMENT* h_B = (INPUT_ELEMENT*)malloc(SIZE_B);
    initInputMatrix(h_B, ROWS_B * COLS_B);
    OUTPUT_ELEMENT* h_C = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    initOutputMatrix(h_C, ROWS_C * COLS_C);
    OUTPUT_ELEMENT* h_C_orig = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    memcpy(h_C_orig, h_C, SIZE_C);
    if (DEBUG_OUTPUT) {
        printf("ALPHA: %f\n", ALPHA);
        printf("BETA: %f\n", BETA);
        printf("A:\n");
        printInputMat(h_A, ROWS_A, COLS_A);
        printf("B:\n");
        printInputMat(h_B, ROWS_B, COLS_B);
        printf("C:\n");
        printOutputMat(h_C, ROWS_C, COLS_C);
    }

    // Device alloc
    INPUT_ELEMENT* d_A;
    cudaMalloc(&d_A, SIZE_A);
    INPUT_ELEMENT* d_B;
    cudaMalloc(&d_B, SIZE_B);
    OUTPUT_ELEMENT* d_C;
    cudaMalloc(&d_C, SIZE_C);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, SIZE_C, cudaMemcpyHostToDevice);

    // Explicitly init device so we only count gemm FLOPS
    cudaError_t initResult = cudaInitDevice(0, cudaDeviceScheduleAuto, 0);
    if (initResult != cudaSuccess) {
        printf("Failed to init device\n");
        return 1;
    }
    cudaError_t setResult = cudaSetDevice(0);
    if (setResult != cudaSuccess) {
        printf("Failed to set device\n");
        return 1;
    }

    // Run kernel
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    d_gemm<<<NUM_BLOCKS,NUM_THREADS>>>(d_A, d_B, d_C, ROWS_A, INNER_DIM, COLS_B, ALPHA, BETA, TRANSPOSE_A, TRANSPOSE_B);
    cudaDeviceSynchronize();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // Report performance metrics
    long d_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double eflops = static_cast<double>(TOTAL_NAIVE_FLOPS) / (static_cast<double>(d_elapsed) / 1e6);
    printf("Device done in %lu microseconds (%f eTFLOPS)\n", d_elapsed, eflops / 1e12);
    printf("%f%% of theoretical max\n", eflops / MAX_FLOPS * 100.0);

    // Copy result to host
    cudaMemcpy(h_C, d_C, SIZE_C, cudaMemcpyDeviceToHost);

    if (DEBUG_OUTPUT) {
        printf("h_C after device computation:\n");
        printOutputMat(h_C, ROWS_C, COLS_C);
    }
    if (DO_CPU_VERIFY) {
        // Verify result
        start = chrono::steady_clock::now();
        h_gemm(h_A, h_B, h_C_orig, ROWS_A, INNER_DIM, COLS_B, ALPHA, BETA, TRANSPOSE_A, TRANSPOSE_B);
        end = chrono::steady_clock::now();
        long h_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
        printf(
            "Host done in %lu microseconds (%f eTFLOPS)\n",
            h_elapsed,
            (static_cast<double>(TOTAL_NAIVE_FLOPS) / (static_cast<double>(h_elapsed) / 1e6)) / 1e12
        );
        if (DEBUG_OUTPUT) {
            printf("CPU result:\n");
            printOutputMat(h_C_orig, ROWS_C, COLS_C);
        }
        if (verify(h_C_orig, h_C)) {
            printf("Output correct\n");
        } else {
            printf("===== Output NOT correct =====\n");
        }
        // Report speedup vs CPU
        printf("Device speedup: %fx\n", static_cast<double>(h_elapsed) / static_cast<double>(d_elapsed));
    } else {
        printf("Skipping CPU verification\n");
    }

    // Free everything
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_orig);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
