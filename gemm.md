---
title: GEMM on Jetson
author: Bradley Gannon
date: 2024-07-15
lang: en-US
publish: false
---

**TL;DR:** I wrote a C++ program that demonstrates a general matrix multiply
(GEMM) implementation for the Jetson AGX Orin devkit. The `gemm` program uses
the tensor cores available on that platform and allows for arbitrary use of the
supported input and output types. There are no external software dependencies
aside from CUDA.

# Purpose and Structure

This post is a **literate program**. That means that this file contains a
complete program within it, in addition to explanatory natural language
throughout. All you need to do to get the source code is extract the lines in
all the code blocks and concatenate them. Everything you need is
[here][github], except for the hardware, the `nvcc` compiler, and CUDA.

[github]: https://github.com/Rinthy-AI/gemm

I wrote this code as a part of Rinthy AI, which is a trio of friends interested
in playing with machine learning tech on our own hardware. We split the cost of
a Jetson AGX Orin development kit, and this program is meant to run exclusively
on that platform---although it may run on others with little or no
modification. The program runs a single general matrix multiply (GEMM)
operation with user-specified dimensions and other parameters. It works as an
example of how to interact with the tensor cores on the the devkit's platform
at a lower level than typical libraries.

# Setup

## Definition of GEMM

TODO

## Includes

TODO

```cpp
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <mma.h>
#include <stdlib.h>

using namespace std;
```

## Defines

TODO

| Input           | Output   | `#define`            |
| --------------- | -------- | -------------------- |
| `__half`        | `__half` | `F16_IN_OUT`         |
| `__half`        | `float`  | `F16_IN_F32_OUT`     |
| `unsigned char` | `int`    | `UINT8_IN_INT32_OUT` |
| `signed char`   | `int`    | `INT8_IN_INT32_OUT`  |
| `__nv_bfloat16` | `float`  | `BF16_IN_F32_OUT`    |
| `tf32`          | `float`  | `TF32_IN_F32_OUT`    |
| `double`        | `double` | `F64_IN_OUT`         |
| `u4`            | `int`    | `U4_IN_INT32_OUT`    |
| `s4`            | `int`    | `S4_IN_INT32_OUT`    |
| `b1`            | `int`    | `B1_IN_INT32_OUT`    |

See [this table][nvidia-in-out-types].

[nvidia-in-out-types]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes

```cpp
#define UINT8_IN_INT32_OUT
```

| WMMA Dimensions (m-n-k) | `#define`      |
|-------------------------|----------------|
| 16 × 16 ×  16           | `MNK_16x16x16` |
| 32 ×  8 ×  16           | `MNK_32x8x16`  |
|  8 × 32 ×  16           | `MNK_8x32x16`  |
| 16 × 16 ×   8           | `MNK_16x16x8`  |
|  8 ×  8 ×   4           | `MNK_8x8x4`    |
|  8 ×  8 ×  32           | `MNK_8x8x32`   |
|  8 ×  8 × 128           | `MNK_8x8x128`  |

Note that not all dimensions are supported for all element types. See link
above.

```cpp
#define MNK_16x16x16

#if defined(F64_IN_OUT)
    #define VERIFY_TOLERANCE 0.005
    typedef double INPUT_ELEMENT;
    typedef double OUTPUT_ELEMENT;
    const double MAX_OPS = 5.3e12/64.0;
    #ifndef MNK_8x8x4
        #error "Selected WMMA dimensions are not supported for F64_IN_OUT"
    #endif
#elif defined(UINT8_IN_INT32_OUT)
    #define VERIFY_TOLERANCE 1
    typedef unsigned char INPUT_ELEMENT;
    typedef int OUTPUT_ELEMENT;
    const double MAX_OPS = 275e12;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for UINT8_IN_INT32_OUT"
    #endif
#else
    #error "Selected matrix element type is not supported"
#endif

#if defined(MNK_16x16x16)
    const unsigned long WMMA_N = 16;
    const unsigned long WMMA_M = 16;
    const unsigned long WMMA_K = 16;
#elif defined(MNK_8x8x4)
    const unsigned long WMMA_N = 8;
    const unsigned long WMMA_M = 8;
    const unsigned long WMMA_K = 4;
#endif

#define IDX(x, y, y_max) x * y_max + y
```

## Constants

TODO

```cpp
const bool          DO_CPU_VERIFY = false;
const bool          DEBUG_OUTPUT  = false;
#if defined(F64_IN_OUT)
    const INPUT_ELEMENT ALPHA     = 1.234;
    const INPUT_ELEMENT BETA      = 5.678;
#elif defined(UINT8_IN_INT32_OUT)
    const INPUT_ELEMENT ALPHA     = 2;
    const INPUT_ELEMENT BETA      = 3;
#endif
const unsigned long ROWS_A        = 4096;
const unsigned long COLS_A        = 4096;
const unsigned long ROWS_B        = COLS_A;
const unsigned long INNER_DIM     = COLS_A;
const unsigned long COLS_B        = 4096;
const unsigned long ROWS_C        = ROWS_A;
const unsigned long COLS_C        = COLS_B;
const unsigned long ROWS_OUT      = ROWS_A;
const unsigned long COLS_OUT      = COLS_B;
const bool          TRANSPOSE_A   = false;
const bool          TRANSPOSE_B   = false;
const unsigned int  WARP_SIZE     = 32;
const dim3          NUM_BLOCKS      (64,64,1);
const dim3          NUM_THREADS     (WARP_SIZE,4,4);
```

I use the term eTOPS in this file to refer to "effective TOPS", which means
"how fast would you have to do arithmetic operations on the given types with
the naive gemm algorithm to match the observed speed"

```cpp
const unsigned long long TOTAL_NAIVE_OPS = (2 * INNER_DIM - 1) * (ROWS_OUT * COLS_OUT) + (ROWS_OUT * COLS_OUT) * 3;
```

## Forward Declarations

This program has a few utility functions. Their implementation details aren't
important for understanding the GEMM kernel, so you can find them in [the
appendix](#appendix-utility-functions). Still, we have to use [forward
declaration][fwd-decl] to tell the compiler what signatures these functions
have before we call them.

[fwd-decl]: https://en.wikipedia.org/wiki/Forward_declaration

The utility functions and their purposes are:

- `h_gemm` does the same GEMM operation as the kernel, except it does it on the
  CPU instead. When `DO_CPU_VERIFY` is `true`, we call this function with the
same inputs as the kernel so we can make sure the kernel is producing the
correct output. The output is passed to `verify` for actual verification.
- `verify` compares the values of the given matrices element by element to make
  sure that they're equal. For floating point types, a small tolerance is
acceptable. Returns `true` if the matrices are equal and returns `false`
otherwise. Also prints the offending element pair in the `false` case.
- `initMatrix` fills the given matrix with random values. The range and
  distribution of the values are not specified.
- `printMat` prints the given matrix in a form that can be easily copied and
  pasted into a Python REPL. This is useful for verification or inspection with
`numpy`. Only called when `DEBUG_OUTPUT` is `true`.

Note that `initMatrix` and `printMat` are both templates over a generic
`ELEMENT`. This is necessary because `INPUT_ELEMENT` and `OUTPUT_ELEMENT` are
usually different, so we would usually need two different function signatures.
The template syntax lets us avoid code duplication by specifying the type we
want at the call site and telling the compiler to figure out the rest.

```cpp
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
);

bool verify(OUTPUT_ELEMENT* h_solution, OUTPUT_ELEMENT* h_C);

template<typename ELEMENT>
void initMatrix(ELEMENT* mat, int len);

template<typename ELEMENT>
void printMat(ELEMENT* mat, int rows, int cols);
```

# Kernel

TODO

```cpp
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
    // TODO add support for arbitrary matrix dimensions
    using namespace nvcuda::wmma;
    int warp_row = (blockIdx.y * blockDim.z + threadIdx.z) * WMMA_M;
    int warp_col = (blockIdx.x * blockDim.y + threadIdx.y) * WMMA_N;
    int tile_run = inner / WMMA_K;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, row_major> A_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, row_major> B_frag;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, col_major> A_frag_t;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, col_major> B_frag_t;
    fragment<accumulator, WMMA_N, WMMA_M, WMMA_K, OUTPUT_ELEMENT> C_frag;

    load_matrix_sync(C_frag, C + IDX(warp_row, warp_col, c_B), c_B, mem_row_major);
    for (int i = 0; i < C_frag.num_elements; i++) {
        C_frag.x[i] *= beta;
    }
    // TODO not sure if y_max needs changing here
    if (t_A && t_B) {
        for (int i = 0; i < tile_run; i++) {
            load_matrix_sync(A_frag_t, A + IDX(i * WMMA_K, warp_row, inner), inner);
            load_matrix_sync(B_frag_t, B + IDX(warp_col, i * WMMA_K, c_B), c_B);
            for (int i = 0; i < A_frag_t.num_elements; i++) {
                A_frag_t.x[i] *= alpha;
            }
            mma_sync(C_frag, A_frag_t, B_frag_t, C_frag);
        }
    } else if (t_A && !t_B) {
        for (int i = 0; i < tile_run; i++) {
            load_matrix_sync(A_frag_t, A + IDX(i * WMMA_K, warp_row, inner), inner);
            load_matrix_sync(B_frag, B + IDX(i * WMMA_K, warp_col, c_B), c_B);
            for (int i = 0; i < A_frag_t.num_elements; i++) {
                A_frag_t.x[i] *= alpha;
            }
            mma_sync(C_frag, A_frag_t, B_frag, C_frag);
        }
    } else if (!t_A && t_B) {
        for (int i = 0; i < tile_run; i++) {
            load_matrix_sync(A_frag, A + IDX(warp_row, i * WMMA_K, inner), inner);
            load_matrix_sync(B_frag_t, B + IDX(warp_col, i * WMMA_K, c_B), c_B);
            for (int i = 0; i < A_frag.num_elements; i++) {
                A_frag.x[i] *= alpha;
            }
            mma_sync(C_frag, A_frag, B_frag_t, C_frag);
        }
    } else if (!t_A && !t_B) {
        for (int i = 0; i < tile_run; i++) {
            load_matrix_sync(A_frag, A + IDX(warp_row, i * WMMA_K, inner), inner);
            load_matrix_sync(B_frag, B + IDX(i * WMMA_K, warp_col, c_B), c_B);
            for (int i = 0; i < A_frag.num_elements; i++) {
                A_frag.x[i] *= alpha;
            }
            mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
    }
    store_matrix_sync(C + IDX(warp_row, warp_col, c_B), C_frag, c_B, mem_row_major);
}
```

# Main Function

```cpp
int main() {
```

## Host Allocation and Initialization

TODO

```cpp
    assert(NUM_BLOCKS.x * NUM_THREADS.y * WMMA_N * NUM_BLOCKS.y * NUM_THREADS.z * WMMA_M == ROWS_OUT * COLS_OUT);
    size_t SIZE_A = ROWS_A * COLS_A * sizeof(INPUT_ELEMENT);
    printf("A: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_A, COLS_A, sizeof(INPUT_ELEMENT), SIZE_A);
    size_t SIZE_B = ROWS_B * COLS_B * sizeof(INPUT_ELEMENT);
    printf("B: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_B, COLS_B, sizeof(INPUT_ELEMENT), SIZE_B);
    size_t SIZE_C = ROWS_C * COLS_C * sizeof(OUTPUT_ELEMENT);
    printf("C: %lu × %lu elements (%lu B each) => %lu B\n", ROWS_C, COLS_C, sizeof(OUTPUT_ELEMENT), SIZE_C);
    size_t TOTAL_SIZE = SIZE_A + SIZE_B + SIZE_C;
    printf("Total eTOPS: %llu\n", TOTAL_NAIVE_OPS);
    printf("Total bytes: %lu\n", TOTAL_SIZE);
    printf("eTOPS per byte: %f\n", static_cast<double>(TOTAL_NAIVE_OPS) / static_cast<double>(TOTAL_SIZE));
    srand(0);
    INPUT_ELEMENT* h_A = (INPUT_ELEMENT*)malloc(SIZE_A);
    initMatrix<INPUT_ELEMENT>(h_A, ROWS_A * COLS_A);
    INPUT_ELEMENT* h_B = (INPUT_ELEMENT*)malloc(SIZE_B);
    initMatrix<INPUT_ELEMENT>(h_B, ROWS_B * COLS_B);
    OUTPUT_ELEMENT* h_C = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    initMatrix<OUTPUT_ELEMENT>(h_C, ROWS_C * COLS_C);
    OUTPUT_ELEMENT* h_C_orig = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    memcpy(h_C_orig, h_C, SIZE_C);
    if (DEBUG_OUTPUT) {
        #if defined(FP64_IN_OUT)
            printf("ALPHA = %f\n", ALPHA);
            printf("BETA = %f\n", BETA);
        #elif defined(UINT8_IN_INT32_OUT)
            printf("ALPHA = %d\n", ALPHA);
            printf("BETA = %d\n", BETA);
        #endif
        printf("TRANSPOSE_A = %s\n", TRANSPOSE_A ? "true" : "false");
        printf("TRANSPOSE_B = %s\n", TRANSPOSE_B ? "true" : "false");
        printf("A = ");
        printMat<INPUT_ELEMENT>(h_A, ROWS_A, COLS_A);
        printf("B = ");
        printMat<INPUT_ELEMENT>(h_B, ROWS_B, COLS_B);
        printf("C = ");
        printMat<OUTPUT_ELEMENT>(h_C, ROWS_C, COLS_C);
    }
```

## Device Allocation and Initialization

TODO

```cpp
    INPUT_ELEMENT* d_A;
    cudaMalloc(&d_A, SIZE_A);
    INPUT_ELEMENT* d_B;
    cudaMalloc(&d_B, SIZE_B);
    OUTPUT_ELEMENT* d_C;
    cudaMalloc(&d_C, SIZE_C);

    cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, SIZE_C, cudaMemcpyHostToDevice);

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
```

## Kernel Launch and Speed Measurement

TODO

```cpp
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    d_gemm<<<NUM_BLOCKS,NUM_THREADS>>>(d_A, d_B, d_C, ROWS_A, INNER_DIM, COLS_B, ALPHA, BETA, TRANSPOSE_A, TRANSPOSE_B);
    cudaDeviceSynchronize();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    long d_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double etops = static_cast<double>(TOTAL_NAIVE_OPS) / (static_cast<double>(d_elapsed) / 1e6);
    printf("Device done in %lu microseconds (%f eTOPS)\n", d_elapsed, etops / 1e12);
    printf("%f%% of theoretical max\n", etops / MAX_OPS * 100.0);
```

## Checking the Result

TODO

```cpp
    if (DEBUG_OUTPUT) {
        printf("h_C after device computation:\n");
        printMat<OUTPUT_ELEMENT>(h_C, ROWS_C, COLS_C);
    }
    if (DO_CPU_VERIFY) {
        cudaMemcpy(h_C, d_C, SIZE_C, cudaMemcpyDeviceToHost);

        start = chrono::steady_clock::now();
        h_gemm(h_A, h_B, h_C_orig, ROWS_A, INNER_DIM, COLS_B, ALPHA, BETA, TRANSPOSE_A, TRANSPOSE_B);
        end = chrono::steady_clock::now();
        long h_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
        printf(
            "Host done in %lu microseconds (%f eTOPS)\n",
            h_elapsed,
            (static_cast<double>(TOTAL_NAIVE_OPS) / (static_cast<double>(h_elapsed) / 1e6)) / 1e12
        );
        if (DEBUG_OUTPUT) {
            printf("CPU result:\n");
            printMat<OUTPUT_ELEMENT>(h_C_orig, ROWS_C, COLS_C);
        }
        if (verify(h_C_orig, h_C)) {
            printf("Output correct\n");
        } else {
            printf("===== Output NOT correct =====\n");
        }
        printf("Device speedup: %fx\n", static_cast<double>(h_elapsed) / static_cast<double>(d_elapsed));
    } else {
        printf("Skipping CPU verification\n");
    }
```

## Cleaning Up

TODO

```cpp
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_orig);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

# Appendix: Utility Functions

These are the definitions of the utility functions that we declared
[above](#forward-declarations). For the purposes of these functions, see that
section.

## `h_gemm`

```cpp
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
```

This is a naive nested `for` loop matrix multiplication. There's no need to be
fancy here. In fact, in this case it's better if everything is a little boring
because then it's easier to prove that it's correct. High confidence in
correctness is useful here because we're depending on this function to tell us
if our kernel code is right.

For every output element, we first scale the existing value by `beta`. (Recall
that in GEMM we accumulate the result into `C`, which already has arbitrary
values in it.) Then, we do a dot product of the corresponding row in `A` and
the corresponding column in `B`. We have to account for transposition of both
inputs, but that can be done easily by swapping the arguments to the `IDX`
macro. Finally, we scale the result by `alpha` and accumulate it into `C`.

```cpp
    INPUT_ELEMENT a, b;
    for (int row = 0; row < r_A; row++) {
        for (int col = 0; col < c_B; col++) {
            C[IDX(row, col, c_B)] *= beta;
            for (int offset = 0; offset < inner; offset++) {
                a = t_A ? A[IDX(offset, row, inner)] : A[IDX(row, offset, inner)];
                b = t_B ? B[IDX(col, offset, c_B)] : B[IDX(offset, col, c_B)];
                C[IDX(row, col, c_B)] += alpha * a * b;
            }
        }
    }
}
```

## `verify`

Given two matrices, check whether the second one (`h_C`) is equal to the first
one (`h_solution`) within some tolerance. If a mismatch appears, print it out
and return `false`. Otherwise, return `true`. For integer types, the tolerance
is `0`, but it's nonzero for floating point types because of the possibility of
[accumulated error][flop-error]. We don't need to know the dimensions of each
matrix because this function should only operate on the output matrix `C`. The
dimensions of `C` are stored in constants `ROWS_C` and `COLS_C`, so we use
those directly. We need to supply different strings to `printf` in the mismatch
case because different output types require different [conversion
specifiers][conv-spec].

[flop-error]: https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems
[conv-spec]: https://en.cppreference.com/w/cpp/io/c/fprintf

```cpp
bool verify(OUTPUT_ELEMENT* h_solution, OUTPUT_ELEMENT* h_C) {
    for (int idx = 0; idx < ROWS_C * COLS_C; idx++) {
        if (abs(h_C[idx] - h_solution[idx]) > VERIFY_TOLERANCE) {
            printf(
                #if defined(F64_IN_OUT)
                    "Found output mismatch at (%lu,%lu): device returned %f but solution is %f\n",
                #elif defined(UINT8_IN_INT32_OUT)
                    "Found output mismatch at (%lu,%lu): device returned %d but solution is %d\n",
                #endif
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
```

## `initMatrix`

Nothing special here. Just loop over all of the elements in the array and
choose a [`rand`][rand]om value for each one. We use the modulo operator to
keep the values between `0` and `16`. As long as the generic `ELEMENT` supports
casting from `int`, then we're all good. Note that we don't need to care about
the fact that the input is actually a matrix. Since the elements are stored as
a contiguous sequence in memory, the dimensionality doesn't matter as long as
we touch all the elements and don't overrun. We depend on the caller to give us
the right value of `len`, which should be the product of the dimensions.

[rand]: https://cplusplus.com/reference/cstdlib/rand/

```cpp
template<typename ELEMENT>
void initMatrix(ELEMENT* mat, int len) {
    for (int idx = 0; idx < len; idx++) {
        mat[idx] = static_cast<ELEMENT>(rand() % 16);
    }
}
```

## `printMat`

Print the contents of the supplied matrix in a way that can be copied and
pasted into a Python REPL. This means that the whole matrix and each row must
be surrounded by `[]` and the elements and rows must be separated by `,`. We
loop over the rows and colums, emitting characters and elements as necessary to
satisfy these requirements. As with `verify` above, different `printf` calls
are needed depending on the type.

```cpp
template<typename ELEMENT>
void printMat(ELEMENT* mat, int rows, int cols) {
    printf("[");
    for (int r = 0; r < rows; r++) {
        printf("[");
        for (int c = 0; c < cols; c++) {
            #if defined(F64_IN_OUT)
                printf("%f,", mat[r * rows + c]);
            #elif defined(UINT8_IN_INT32_OUT)
                printf("%d,", mat[r * rows + c]);
            #endif
        }
        printf("],");
        if (r != rows - 1) {
            printf("\n");
        }
    }
    printf("]\n");
}
```
