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
all the code blocks and concatenate them. To run the code, clone [the
repo][github] and run `make`. You need `nvcc` and CUDA. The compiled binary
will be in the repo root and is called `gemm`. To play with other input and
output types, matrix dimensions, transposition, and other parameters, change
the `#define`s and constants in the code according to the descriptions below.

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

GEMM stands for general matrix multiply. It is one of the Level 3 [Basic Linear
Algebra Subprograms][blas] (BLAS) and takes the form

[blas]: https://dl.acm.org/doi/pdf/10.1145/77626.79170

$$\mathbf{C} \leftarrow \alpha\mathbf{A}\mathbf{B} + \beta\mathbf{C}$$

where $\mathbf{A}$, $\mathbf{B}$, and $\mathbf{C}$ are real matrices and
$\alpha$ and $\beta$ are real scalars. $\mathbf{A}$ and/or $\mathbf{B}$ may be
transposed at the user's option.

Note that $\mathbf{C}$ appears on both sides of the assignment operator. This
means that the existing values of $\mathbf{C}$ must be a part of the inputs to
the GEMM algorithm *and also* the outputs must be stored in $\mathbf{C}$. In
practice, to avoid intermediate allocations we have to compute
$\beta\mathbf{C}$ first and then accumulate the results of
$\alpha\mathbf{A}\mathbf{B}$ into $\mathbf{C}$ second.

I chose GEMM as the goal for this program because of its usefulness in machine
learning. A fast GEMM implementation---or at least one that teaches me
about the hardware and programming models---has value for any future work we
might do in this area.

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

TODO

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
```

TODO

```cpp
#if defined(MNK_16x16x16)
    const unsigned long WMMA_N = 16;
    const unsigned long WMMA_M = 16;
    const unsigned long WMMA_K = 16;
#elif defined(MNK_8x8x4)
    const unsigned long WMMA_N = 8;
    const unsigned long WMMA_M = 8;
    const unsigned long WMMA_K = 4;
#endif
```

TODO

```cpp
#define TRANSPOSE_A false
#define TRANSPOSE_B false
```

TODO

```cpp
#define IDX(x, y, y_max) x * y_max + y
```

TODO

```cpp
#define RND_UP_MULT(n, m) (n % m == 0) ? ((n / m) * m) : ((n / m + 1) * m)
```

## Constants

TODO

```cpp
const bool          DO_CPU_VERIFY    = false;
const bool          DEBUG_OUTPUT     = false;
#if defined(F64_IN_OUT)
    const INPUT_ELEMENT ALPHA        = 1.234;
    const INPUT_ELEMENT BETA         = 5.678;
#elif defined(UINT8_IN_INT32_OUT)
    const INPUT_ELEMENT ALPHA        = 2;
    const INPUT_ELEMENT BETA         = 3;
#endif
const unsigned long ROWS_A           = 4096;
const unsigned long ROWS_A_PADDED    = RND_UP_MULT(ROWS_A, WMMA_M);
const unsigned long COLS_A           = 4096;
const unsigned long COLS_A_PADDED    = RND_UP_MULT(COLS_A, WMMA_K);
const unsigned long ROWS_B           = COLS_A;
const unsigned long ROWS_B_PADDED    = COLS_A_PADDED;
const unsigned long INNER_DIM        = COLS_A;
const unsigned long INNER_DIM_PADDED = COLS_A_PADDED;
const unsigned long COLS_B           = 4096;
const unsigned long COLS_B_PADDED    = RND_UP_MULT(COLS_B, WMMA_N);
const unsigned long ROWS_C           = ROWS_A;
const unsigned long ROWS_C_PADDED    = ROWS_A_PADDED;
const unsigned long COLS_C           = COLS_B;
const unsigned long COLS_C_PADDED    = COLS_B_PADDED;
const unsigned int  WARP_SIZE        = 32;
const dim3          NUM_BLOCKS         (64,64,1);
const dim3          NUM_THREADS        (WARP_SIZE,4,4);
```

TODO

I use the term eTOPS in this file to refer to "effective TOPS", which means
"how fast would you have to do arithmetic operations on the given types with
the naive gemm algorithm to match the observed speed"

```cpp
const unsigned long long TOTAL_NAIVE_OPS = (2 * INNER_DIM - 1) * (ROWS_C * COLS_C)
                                         + (ROWS_C * COLS_C) * 3;
```

## Forward Declarations

This program has a few utility functions. Their implementation details aren't
important for understanding the GEMM kernel, so you can find the full
definitions in [the appendix](#appendix-utility-functions). Still, we have to
use [forward declaration][fwd-decl] to tell the compiler what signatures these
functions have before we call them.

[fwd-decl]: https://en.wikipedia.org/wiki/Forward_declaration

<details>
<summary>Click to show/hide forward declarations</summary>

```cpp
void h_gemm(
    INPUT_ELEMENT* A,
    INPUT_ELEMENT* B,
    OUTPUT_ELEMENT* C,
    unsigned long r_A,
    unsigned long inner,
    unsigned long c_B,
    INPUT_ELEMENT alpha,
    INPUT_ELEMENT beta
);

bool verify(OUTPUT_ELEMENT* h_solution, OUTPUT_ELEMENT* h_C);

template<typename ELEMENT>
void initMatrix(ELEMENT* mat, int len);

template<typename ELEMENT>
void printMat(ELEMENT* mat, int rows, int cols);
```

</details>

# Kernel

This is the code that actually runs on the GPU. Let's examine the function
signature first.

```cpp
__global__ void d_gemm(
    INPUT_ELEMENT* A,
    INPUT_ELEMENT* B,
    OUTPUT_ELEMENT* C,
    unsigned long r_A,
    unsigned long inner,
    unsigned long c_B,
    INPUT_ELEMENT alpha,
    INPUT_ELEMENT beta
) {
```

TODO

```cpp
    using namespace nvcuda::wmma;
    int warp_row = (blockIdx.x * blockDim.y + threadIdx.y) * WMMA_M;
    int warp_col = (blockIdx.y * blockDim.z + threadIdx.z) * WMMA_N;
    int tile_run = inner / WMMA_K;
```

TODO

```cpp
    #if TRANSPOSE_A
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, col_major> A_frag;
        #define IDX_A IDX(i * WMMA_K, warp_row, inner)
    #else
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, row_major> A_frag;
        #define IDX_A IDX(warp_row, i * WMMA_K, inner)
    #endif
    #if TRANSPOSE_B
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, col_major> B_frag;
        #define IDX_B IDX(warp_col, i * WMMA_K, c_B)
    #else
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, INPUT_ELEMENT, row_major> B_frag;
        #define IDX_B IDX(i * WMMA_K, warp_col, c_B)
    #endif
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, OUTPUT_ELEMENT> C_frag;
```

TODO

```cpp
    load_matrix_sync(C_frag, C + IDX(warp_row, warp_col, c_B), c_B, mem_row_major);
    for (int i = 0; i < C_frag.num_elements; i++) {
        C_frag.x[i] *= beta;
    }
```

TODO

```cpp
    for (int i = 0; i < tile_run; i++) {
        load_matrix_sync(A_frag, A + IDX_A, inner);
        load_matrix_sync(B_frag, B + IDX_B, c_B);
        for (int i = 0; i < A_frag.num_elements; i++) {
            A_frag.x[i] *= alpha;
        }
        mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
```

TODO

```cpp
    store_matrix_sync(C + IDX(warp_row, warp_col, c_B), C_frag, c_B, mem_row_major);
}
```

# Main Function

Now we come to the entry point of the program.

```cpp
int main() {
```

## Host Allocation and Initialization

First, we check that the dimensions of the blocks and grids match the number of
elements in the output matrix. It's easy to get these wrong because reasoning
about all the different loops and dimensions is often difficult. This runtime
check catches a decent number of errors in this category.

We assume that `NUM_BLOCKS.z` is equal to `1`, which implies a two-dimensional
grid. We also assume that `NUM_THREADS.x` is equal to `WARP_SIZE`. The tensor
cores work most efficiently when the maximum number of warps per scheduler are
executing the same instruction, so we encourage that by setting that as one of
the block dimensions. We know that the dimensions of a single tensor core
output matrix are equal to `(WMMA_N, WMMA_M)` elements. Both of those values
come straight from Nvidia's documentation and are `#define`d above. All we need
to do is multiply the number of elements per output tile by the number of
tiles, and we'll get the number of output elements for this configuration. That
value should always match the number of elements in the actual output array.

```cpp
    assert(
        NUM_BLOCKS.x * NUM_THREADS.y * WMMA_N
        * NUM_BLOCKS.y * NUM_THREADS.z * WMMA_M
        == ROWS_C_PADDED * COLS_C_PADDED
    );
```

To allocate memory for our three matrices, we need to know their sizes in
bytes. We'll compute those values now by finding the number of elements and
then multiplying that value by the number of bytes per element. We also print
the dimensions and sizes of the matrices.

```cpp
    size_t SIZE_A = ROWS_A * COLS_A * sizeof(INPUT_ELEMENT);
    printf(
        "A original   : %5lu × %5lu elements, %lu B => %9lu B\n",
        ROWS_A,
        COLS_A,
        sizeof(INPUT_ELEMENT),
        SIZE_A
    );
    size_t SIZE_A_PADDED = ROWS_A_PADDED * COLS_A_PADDED * sizeof(INPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu elements, %lu B => %9lu B\n",
        ROWS_A_PADDED,
        COLS_A_PADDED,
        sizeof(INPUT_ELEMENT),
        SIZE_A_PADDED
    );
    size_t SIZE_B = ROWS_B * COLS_B * sizeof(INPUT_ELEMENT);
    printf(
        "B original   : %5lu × %5lu elements, %lu B => %9lu B\n",
        ROWS_B,
        COLS_B,
        sizeof(INPUT_ELEMENT),
        SIZE_B
    );
    size_t SIZE_B_PADDED = ROWS_B_PADDED * COLS_B_PADDED * sizeof(INPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu elements, %lu B => %9lu B\n",
        ROWS_B_PADDED,
        COLS_B_PADDED,
        sizeof(INPUT_ELEMENT),
        SIZE_B_PADDED
    );
    size_t SIZE_C = ROWS_C * COLS_C * sizeof(OUTPUT_ELEMENT);
    printf(
        "C original   : %5lu × %5lu elements, %lu B => %9lu B\n",
        ROWS_C,
        COLS_C,
        sizeof(OUTPUT_ELEMENT),
        SIZE_C
    );
    size_t SIZE_C_PADDED = ROWS_C_PADDED * COLS_C_PADDED * sizeof(OUTPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu elements, %lu B => %9lu B\n\n",
        ROWS_C_PADDED,
        COLS_C_PADDED,
        sizeof(OUTPUT_ELEMENT),
        SIZE_C_PADDED
    );
```

Now we print some information about the operation we're about to run. It's
useful to know the [arithmetic intensity][arith-intens] of the computation,
which is the ratio of the total number of math operations to the total number
of bytes that need to move during the computation. This figure gives some
insight into whether or not the computation will be limited by the compute
performance or memory bandwidth of the target platform.

[arith-intens]: https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity

```cpp
    size_t TOTAL_SIZE = SIZE_A + SIZE_B + SIZE_C;
    printf("Total eOPS   : %12llu\n", TOTAL_NAIVE_OPS);
    printf("Input size   : %12lu B\n", SIZE_A + SIZE_B);
    printf("Output size  : %12lu B\n", SIZE_C);
    printf(
        "eOPS per byte: %12f\n\n",
        static_cast<double>(TOTAL_NAIVE_OPS) / static_cast<double>(TOTAL_SIZE)
    );
```

Set the random seed to a fixed value so we always get the same element values
for a given configuration. This eliminates a source of random variation in
performance.

```cpp
    srand(0);
```

Now we allocate memory for `A`, `B`, and `C` and initialize them to random
values. Note our use of `INPUT_ELEMENT` and `OUTPUT_ELEMENT` in the calls to
`initMatrix`, which tells the compiler to generate function implementations for
those specific types using the template defined elsewhere.

```cpp
    INPUT_ELEMENT* h_A = (INPUT_ELEMENT*)malloc(SIZE_A);
    initMatrix<INPUT_ELEMENT>(h_A, ROWS_A * COLS_A);
    INPUT_ELEMENT* h_B = (INPUT_ELEMENT*)malloc(SIZE_B);
    initMatrix<INPUT_ELEMENT>(h_B, ROWS_B * COLS_B);
    OUTPUT_ELEMENT* h_C = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    initMatrix<OUTPUT_ELEMENT>(h_C, ROWS_C * COLS_C);
```

The tensor cores do fast matrix multiplications, but they only operate on
inputs and outputs with fixed dimensions (`WMMA_M`, `WMMA_N`, and `WMMA_K`). If
our input matrices `A` and `B` have dimensions that aren't multiples of the
corresponding tensor core sizes, then we're going to end up missing the "extra"
values that sit outside the last tiles. We can't just run for an extra tile to
get those values because we'd end up reading values from beginnings of the
*next* rows and exceed the boundaries of the matrices on the last rows.
Instead, we need to allocate padded versions of the matrices with dimensions
that *are* a multiple of the tensor core dimensions and fill the unneeded
values with zeros. Then, after the kernel is done running, we'll need to
extract the values we want from the output to get the true result.

```cpp
    INPUT_ELEMENT* h_A_padded = (INPUT_ELEMENT*)malloc(SIZE_A_PADDED);
    INPUT_ELEMENT* h_B_padded = (INPUT_ELEMENT*)malloc(SIZE_B_PADDED);
    OUTPUT_ELEMENT* h_C_padded = (OUTPUT_ELEMENT*)malloc(SIZE_C_PADDED);
```

TODO

```cpp
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    for (int r = 0; r < ROWS_A_PADDED; r++) {
        memcpy(
            h_A_padded + IDX(r, 0, COLS_A_PADDED),
            h_A + IDX(r, 0, COLS_A),
            COLS_A * sizeof(INPUT_ELEMENT)
        );
        for (int c = COLS_A; c < COLS_A_PADDED; c++) {
            h_A_padded[IDX(r, c, COLS_A_PADDED)] = static_cast<INPUT_ELEMENT>(0);
        }
    }
    for (int r = 0; r < ROWS_B_PADDED; r++) {
        memcpy(
            h_B_padded + IDX(r, 0, COLS_B_PADDED),
            h_B + IDX(r, 0, COLS_B),
            COLS_B * sizeof(INPUT_ELEMENT)
        );
        for (int c = COLS_B; c < COLS_B_PADDED; c++) {
            h_B_padded[IDX(r, c, COLS_B_PADDED)] = static_cast<INPUT_ELEMENT>(0);
        }
    }
    for (int r = 0; r < ROWS_C_PADDED; r++) {
        memcpy(
            h_C_padded + IDX(r, 0, COLS_C_PADDED),
            h_C + IDX(r, 0, COLS_C),
            COLS_C * sizeof(OUTPUT_ELEMENT)
        );
        for (int c = COLS_C; c < COLS_C_PADDED; c++) {
            h_C_padded[IDX(r, c, COLS_C_PADDED)] = static_cast<OUTPUT_ELEMENT>(0);
        }
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    unsigned long pad_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    printf("Padding      : %12lu μs\n\n", pad_elapsed);
```

If we're going to do CPU verification later, then we need an original copy of
the `C` matrix. GEMM modifies `C` in the general case, so running the kernel
changes the values in `h_C` after copying them back from the GPU.  We'll save a
copy of the original values in `h_C_orig`. We don't actually need to do this
unless `DO_CPU_VERIFY` is `true`, but adding the extra logic to handle
conditionally allocating and freeing the memory isn't worth the benefit in my
opinion, so we just do it every time.

```cpp
    OUTPUT_ELEMENT* h_C_orig = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    memcpy(h_C_orig, h_C, SIZE_C);
```

If `DEBUG_OUTPUT` is `true`, then we'll print whether `A` and `B` will be
transposed during the operation, as well as the initial contents of `A`, `B`,
and `C`. The `printMat` function is generic over the element type---just like
`initMatrix` above---and prints the matrix elements in a format that can be
easily copied into a Python REPL.

```cpp
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

The inputs are allocated and initialized on the CPU, but to do the GEMM
operation we need to move them to the GPU. CUDA has functions in its API for
handling this.

```cpp
    INPUT_ELEMENT* d_A;
    cudaMalloc(&d_A, SIZE_A_PADDED);
    INPUT_ELEMENT* d_B;
    cudaMalloc(&d_B, SIZE_B_PADDED);
    OUTPUT_ELEMENT* d_C;
    cudaMalloc(&d_C, SIZE_C_PADDED);

    cudaMemcpy(d_A, h_A_padded, SIZE_A_PADDED, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_padded, SIZE_B_PADDED, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_padded, SIZE_C_PADDED, cudaMemcpyHostToDevice);
```

We're almost ready. If we were to launch the kernel now, we might accidentally
include the device initialization in our speed measurement. We can avoid that
by initializing the device explicitly now.

```cpp
    cudaError_t initResult = cudaInitDevice(0, cudaDeviceScheduleAuto, 0);
    if (initResult != cudaSuccess) {
        printf("Failed to init device\n");
        return 1;
    }
```

## Kernel Launch and Speed Measurement

We're finally done setting up. We can launch the kernel and measure the time it
takes to execute. The call to `cudaDeviceSynchronize` guarantees that the GPU
will finish its most recent kernel before advancing to the next line. Normally,
CPU execution continues after a kernel launch---which is the point of having a
GPU at all---and that's usually the desired behavior. In this case, we *want*
to block on the kernel's execution because we're trying to measure how long it
takes to run.

```cpp
    start = chrono::steady_clock::now();
    d_gemm<<<NUM_BLOCKS,NUM_THREADS>>>(
        d_A,
        d_B,
        d_C,
        ROWS_A_PADDED,
        INNER_DIM_PADDED,
        COLS_B_PADDED,
        ALPHA,
        BETA
    );
    cudaDeviceSynchronize();
    end = chrono::steady_clock::now();
```

Get the number of microseconds between the start and end of the kernel's
runtime, and then report some information about its effective speed.

```cpp
    long d_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double eops = static_cast<double>(TOTAL_NAIVE_OPS) / (static_cast<double>(d_elapsed) / 1e6);
    printf("Device       : %12lu μs (%11f eTOPS)\n", d_elapsed, eops / 1e12);
```

TODO

```cpp
    cudaMemcpy(h_C_padded, d_C, SIZE_C_PADDED, cudaMemcpyDeviceToHost);
    start = chrono::steady_clock::now();
    for (int r = 0; r < ROWS_C; r++) {
        memcpy(
            h_C + IDX(r, 0, COLS_C),
            h_C_padded + IDX(r, 0, COLS_C_PADDED),
            COLS_C * sizeof(OUTPUT_ELEMENT)
        );
    }
    end = chrono::steady_clock::now();
    unsigned long extract_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
```

## Checking the Result

TODO

```cpp
    if (DEBUG_OUTPUT) {
        printf("h_C after device computation:\n");
        printMat<OUTPUT_ELEMENT>(h_C, ROWS_C, COLS_C);
    }
    if (DO_CPU_VERIFY) {
        start = chrono::steady_clock::now();
        h_gemm(h_A, h_B, h_C_orig, ROWS_A, INNER_DIM, COLS_B, ALPHA, BETA);
        end = chrono::steady_clock::now();
        unsigned long h_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
        printf(
            "Host         : %12lu μs (%11f eTOPS)\n\n",
            h_elapsed,
            (static_cast<double>(TOTAL_NAIVE_OPS) / (static_cast<double>(h_elapsed) / 1e6)) / 1e12
        );
        if (DEBUG_OUTPUT) {
            printf("CPU result:\n");
            printMat<OUTPUT_ELEMENT>(h_C_orig, ROWS_C, COLS_C);
        }
        if (verify(h_C_orig, h_C)) {
            printf("Extraction   : %12lu μs\n\n", extract_elapsed);
            printf("Output correct\n");
            printf(
                "Device speedup: %fx\n",
                static_cast<double>(h_elapsed) / static_cast<double>(d_elapsed)
            );
        } else {
            printf("===== Output NOT correct =====\n");
        }
    } else {
        printf("Skipping CPU verification\n");
    }
    printf("%f%% of theoretical max\n", eops / MAX_OPS * 100.0);
```

## Cleaning Up

We've done what we came here to do. All we really need to do now is return
zero, but it doesn't hurt to deallocate all of the memory we've been using. I
think the operating system would take care of this for us after the process
terminates, but for completeness we might as well do it ourselves.

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
[above](#forward-declarations).

Note that `initMatrix` and `printMat` are both templates over a generic
`ELEMENT`. This is necessary because `INPUT_ELEMENT` and `OUTPUT_ELEMENT` are
usually different, so we would usually need two different function signatures.
The template syntax lets us avoid code duplication by specifying the type we
want at the call site and telling the compiler to figure out the rest.

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
    INPUT_ELEMENT beta
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

This function is only called when `DO_CPU_VERIFY` is `true`. For large
matrices, this takes a long time.

```cpp
    INPUT_ELEMENT a, b;
    for (int row = 0; row < r_A; row++) {
        for (int col = 0; col < c_B; col++) {
            C[IDX(row, col, c_B)] *= beta;
            for (int offset = 0; offset < inner; offset++) {
                #if TRANSPOSE_A
                    a = A[IDX(offset, row, inner)];
                #else
                    a = A[IDX(row, offset, inner)];
                #endif
                #if TRANSPOSE_B
                    b = B[IDX(col, offset, c_B)];
                #else
                    b = B[IDX(offset, col, c_B)];
                #endif
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
                idx / COLS_C,
                idx % COLS_C,
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

This function is only called when `DEBUG_OUTPUT` is `true`. For large matrices,
this produces a lot of output.

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
