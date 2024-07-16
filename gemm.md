---
title: GEMM on Jetson
author: Bradley Gannon
date: 2024-07-22
lang: en-US
publish: false
---

**TL;DR:** I wrote a C++ program that demonstrates a general matrix multiply
(GEMM) implementation for the [Jetson AGX Orin devkit][devkit]. The `gemm`
program uses the [tensor cores][tensor-cores] available on that platform and
allows for arbitrary use of the supported input and output types (except
[sub-byte types][sub-byte]). There are no external software dependencies aside
from CUDA.  The performance is nowhere near cuBLAS, but the code may be useful
as a reference.

[devkit]: https://developer.nvidia.com/embedded/jetson-developer-kits
[tensor-cores]: https://www.nvidia.com/en-us/data-center/tensor-cores/
[sub-byte]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations

# Purpose and Structure

This post is a **literate program**. That means that this file contains a
complete program within it, in addition to explanatory natural language
throughout. All you need to do to get the source code is extract the lines in
all the code blocks and concatenate them. To run the code, clone [the
repo][github] and run `make`. You need `nvcc`, CUDA, and Python. The compiled
binary will be in the repo root and is called `gemm`. To play with other input
and output types, matrix dimensions, transposition, and other parameters,
change the `#define`s and constants in the code according to the descriptions
below.

[github]: https://github.com/Rinthy-AI/gemm

I wrote this code as a part of Rinthy AI, which is a trio of friends interested
in playing with machine learning tech on our own hardware. We split the cost of
a Jetson AGX Orin development kit, and this program is meant to run exclusively
on that platform—although it may run on others with little or no modification.
The program runs a single general matrix multiply (GEMM) operation with
user-specified dimensions and other parameters. It works as an example of how
to interact with the tensor cores on the the devkit's platform at a lower level
than typical libraries.

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
learning. Understanding some CUDA and how GEMM works with the tensor cores has
value for any future work we might do in this area.

## Includes

The first thing we have to do is pull in some dependencies. `cassert`,
`chrono`, `iostream`, and `stdlib.h` are all for basic I/O, timing, and other
stuff related to the test harness rather than the kernel itself. We need
`cuda.h` for any CUDA stuff to work at all, and we also need `cuda_bf16.h` so
we can use the [`bfloat16`][bf16] type, which is a variant of half-precision
floating point. `mma.h` contains everything we need to interact with the tensor
cores and stands for "matrix multiply accumulate".

[bf16]: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus

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

Here we define our inputs and outputs. The tensor cores [are
modelled][tc-model] as special coprocessing units that perform the operations
$\mathbf{D} = \mathbf{A}\mathbf{B} + \mathbf{C}$ or $\mathbf{C} =
\mathbf{A}\mathbf{B} + \mathbf{C}$. The tensor cores only support a fixed set
of input and output types, where $\mathbf{A}$ and $\mathbf{B}$ are considered
inputs and $\mathbf{C}$ and $\mathbf{D}$ are outputs. To choose a particular
input/output type pair, we `#define` a corresponding identifier. The supported
type pairs and their corresponding identifiers are listed in the table
below,[^sub-byte-support] which I have copied from [here][types-table].

[tc-model]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-matrix-functions
[types-table]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes

[^sub-byte-support]: The tensor cores support additional type pairs where the
inputs are less than one byte wide. They are supported on an experimental basis
and aren't that interesting to me, so I didn't implement them here. Some
interesting extra operations are also supported for these types.

| Input           | Output   | `#define`         |
| --------------- | -------- | ------------------|
| `half`          | `float`  | `F16_IN_F32_OUT`  |
| `half`          | `half`   | `F16_IN_OUT`      |
| `unsigned char` | `int`    | `U8_IN_I32_OUT`   |
| `signed char`   | `int`    | `I8_IN_I32_OUT`   |
| `nv_bfloat16`   | `float`  | `BF16_IN_F32_OUT` |
| `tf32`          | `float`  | `TF32_IN_F32_OUT` |
| `double`        | `double` | `F64_IN_OUT`      |

Below we select the type pair that takes signed 8-bit integers as input and
accumulates into signed 32-bit integers on the output. You can select a
different type pair by replacing the `#define`.

```cpp
#define I8_IN_I32_OUT
```

Each type pair also supports a fixed set of matrix dimensions. In the kernel,
it will be up to us to load these "tiles" in the correct order by indexing into
the given arrays.[^tiling] The supported tile dimensions are listed in the
table linked above, and their mapping to `#define`s is shown in the table
below.

[^tiling]: If you're already familiar with [matmul tiling][tiling-howto], the
tensor cores allow you to do the same thing but *faster*. In other words, using
the tensor cores is more or less like getting a tiled matmul for free.

[tiling-howto]: https://penny-xu.github.io/blog/tiled-matrix-multiplication

| WMMA Dimensions (m-n-k) | `#define`      |
|-------------------------|----------------|
| 16 × 16 ×  16           | `MNK_16x16x16` |
| 32 ×  8 ×  16           | `MNK_32x8x16`  |
|  8 × 32 ×  16           | `MNK_8x32x16`  |
| 16 × 16 ×   8           | `MNK_16x16x8`  |
|  8 ×  8 ×   4           | `MNK_8x8x4`    |

In general, types that are smaller in memory allow for larger tiles. By
default, we select square tiles of 256 elements. Note that in the `MNK`
notation as given by Nvidia, the $\mathbf{A}$ tile has dimensions $M \times K$,
the $\mathbf{B}$ tile has dimensions $K \times N$, and the $\mathbf{C}$ tile
has dimensions $M \times N$.

```cpp
#define MNK_16x16x16
```

We'll need the `MNK` values as individual constants later, so we'll use
preprocessor logic to define them now. For brevity, I've hidden most of the
logic.[^unsigned-long]

[^unsigned-long]: It may seem strange that we're using `unsigned long` for such
small constant values. In fact, you'll see this throughout the program. The
only reason I've done this is to make things easier when computing large
values, such as the number of eOPS in the overall operation or the number of
bytes in an input matrix. If I declared these values as `int` or similar, I'd
have to worry about rollover, but this way I can avoid all that. I don't know
whether or not this affects performance, but I doubt it.

```cpp
#if defined(MNK_16x16x16)
    const unsigned long WMMA_M =  16;
    const unsigned long WMMA_N =  16;
    const unsigned long WMMA_K =  16;
```

<details>
<summary>Click to show/hide `MNK` constant logic</summary>

```cpp
#elif defined(MNK_32x8x16)
    const unsigned long WMMA_M =  32;
    const unsigned long WMMA_N =   8;
    const unsigned long WMMA_K =  16;
#elif defined(MNK_8x32x16)
    const unsigned long WMMA_M =   8;
    const unsigned long WMMA_N =  32;
    const unsigned long WMMA_K =  16;
#elif defined(MNK_16x16x8)
    const unsigned long WMMA_M =  16;
    const unsigned long WMMA_N =  16;
    const unsigned long WMMA_K =   8;
#elif defined(MNK_8x8x4)
    const unsigned long WMMA_M =   8;
    const unsigned long WMMA_N =   8;
    const unsigned long WMMA_K =   4;
#else
    #error "No MNK setting selected"
#endif
```

</details>

Now we define the actual input and output types for our selected type pair
using `typedef`. In all future code, we'll use `INPUT_ELEMENT` and
`OUTPUT_ELEMENT` to refer to the generic types defined here. Later we'll also
optionally check whether or not the kernel gave the correct answer, but we need
to allow some tolerance in certain cases due to floating-point rounding error.
`VERIFY_TOLERANCE` will be the maximum allowable difference between the
accepted answer for any output element and the computed result. For our
selected type pair, we're only dealing with integers, so the answer should be
exact.

Also, it's possible to `#define` a type pair and `MNK` setting that aren't
supported together, so we check for that using the data from the linked table.
I've pulled out the case for our chosen type pair and hidden the rest, since
they're not that interesting.

```cpp
#if defined(I8_IN_I32_OUT)
    #define VERIFY_TOLERANCE 0
    typedef signed char      INPUT_ELEMENT;
    typedef int              OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for I8_IN_I32_OUT"
    #endif
```

<details>
<summary>Click to show/hide other type definitions and tile dimensions checks</summary>

```cpp
#elif defined(F16_IN_F32_OUT)
    #define VERIFY_TOLERANCE 5
    typedef half             INPUT_ELEMENT;
    typedef float            OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for F16_IN_F32_OUT"
    #endif
#elif defined(F16_IN_OUT)
    #define VERIFY_TOLERANCE 50
    typedef half             INPUT_ELEMENT;
    typedef half             OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for F16_IN_OUT"
    #endif
#elif defined(U8_IN_I32_OUT)
    #define VERIFY_TOLERANCE 0
    typedef unsigned char    INPUT_ELEMENT;
    typedef int              OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for U8_IN_I32_OUT"
    #endif
#elif defined(BF16_IN_F32_OUT)
    #define VERIFY_TOLERANCE 10
    typedef nv_bfloat16      INPUT_ELEMENT;
    typedef float            OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x16) && !defined(MNK_32x8x16) && !defined(MNK_8x32x16)
        #error "Selected WMMA dimensions are not supported for BF16_IN_F32_OUT"
    #endif
#elif defined(TF32_IN_F32_OUT)
    #define VERIFY_TOLERANCE 5
    typedef float            INPUT_ELEMENT;
    typedef float            OUTPUT_ELEMENT;
    #if !defined(MNK_16x16x8)
        #error "Selected WMMA dimensions are not supported for TF32_IN_F32_OUT"
    #endif
#elif defined(F64_IN_OUT)
    #define VERIFY_TOLERANCE 0.005
    typedef double           INPUT_ELEMENT;
    typedef double           OUTPUT_ELEMENT;
    #ifndef MNK_8x8x4
        #error "Selected WMMA dimensions are not supported for F64_IN_OUT"
    #endif
#else
    #error "No input/output type selected"
#endif
```

</details>

The definition of GEMM allows the user to transpose $\mathbf{A}$ and/or
$\mathbf{B}$. These `#define`s encode that information. I originally made these
runtime constants, but the code to make it work was hard to follow. It made
more sense to make them compile-time options instead. In an actual application,
this approach might be reasonable because the particular layout of matmuls for
a given model or workload might be known at compile time.

```cpp
#define TRANSPOSE_A false
#define TRANSPOSE_B false
```

All matrices in this program are stored as contiguous arrays in row-major
order. This means that the dimension information is stored separately from the
data itself, and it's on us to index into the array properly. For convenience,
let's `#define` a macro that takes our desired 2D indices and does the indexing
math for us. This makes the code easier to read. We also have to pass `y_max`,
which is the number of elements to skip forward in order to increment `y` by 1.
For row-major matrices, `y_max` is equal to the number of columns in the
matrix. However, as we'll see later, we can play with the inputs to this macro
to get efficient in-place transposition.

Note that everything is over-parenthesized because this is a macro, and the
arguments are *expressions*, not values. If `x` is something like `idx /
NUM_ROWS`, then without parentheses the order of operations might not play out
how we'd expect. (This bit me during development.) To ensure that the arguments
are evaluated first, we wrap them up in parens.

```cpp
#define IDX(x, y, y_max) ((x) * (y_max) + (y))
```

The tensor cores operate on fixed tile sizes. That presents an issue when our
input matrices have dimensions that aren't a multiple of the tile dimensions.
This is a variant of the [tile quantization][tile-quant] problem. We'll solve
it later by padding the inputs with zeros up to the nearest multiples in each
dimension. This macro does that rounding up operation for us. Given some input
`n` and desired factor `m`, this macro checks whether `n` is divisble by `m`.
If it is, then there's nothing to do, so we resolve to `n`. Otherwise, we use
integer division to get the next *lowest* multiple of `m`, add 1, and then
multiply back up by `m`. As above, we parenthesize everything.

[tile-quant]: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#tile-quant

```cpp
#define RND_UP_MULT(n, m) ((n) % (m) == 0) ? (n) : (((n) / (m) + 1) * (m))
```

## Constants

This is the control panel of the program. Aside from the `#define`s above,
these constants encode everything about the inputs to the GEMM operation. Our
kernel will take some of these values as arguments in order to properly
separate it from the rest of the program, but for this purpose it makes sense
to set the values statically.

`DO_CPU_VERIFY` enables a correctness check. In the appendix you'll find a
simple (and slow) CPU implementation of the GEMM operation. When this constant
is `true`, we feed the original inputs to this checking function and compare
its results to the kernel. On the Jetson devkit, square matrices with dimension
1000 take about one second to evaluate on the CPU, so beyond that I turn it
off.

`DEBUG_OUTPUT` causes the program to print the contents of the input and output
matrices. This is useful for debugging but only makes sense for small matrices.

```cpp
const bool          DO_CPU_VERIFY    = false;
const bool          DEBUG_OUTPUT     = false;
```

Here we'll define the values of $\alpha$ and $\beta$. Different type pairs need
different literals, so we use preprocessor logic to make that happen. Note that
the base case covers all floating point type pairs, but for those types with
low precision, the given literals are usually not representable. The compiler
seems to gracefully handle this by adjusting the actual values to the nearest
representable one.

```cpp
#if defined(U8_IN_I32_OUT)
    const INPUT_ELEMENT ALPHA        =      2;
    const INPUT_ELEMENT BETA         =      3;
#elif defined(I8_IN_I32_OUT)
    const INPUT_ELEMENT ALPHA        =     -2;
    const INPUT_ELEMENT BETA         =      3;
#else
    const INPUT_ELEMENT ALPHA        = -1.234;
    const INPUT_ELEMENT BETA         =  5.678;
#endif
```

Now we define the sizes of the matrices. Note that matrix multiplication is
undefined when the number of columns in the left matrix is not equal to the
number of rows in the right matrix. This means that we can only really change
*three* parameters here: the rows of $\mathbf{A}$, the columns of $\mathbf{B}$,
and the shared "inner" dimension. Also, the dimensions of $\mathbf{C}$ are
implied by these values, too. We encode all of these constraints by assigning
subsequent constants based on the earlier three literals.

```cpp
const unsigned long ROWS_A           = 10000;
const unsigned long COLS_A           = 10000;
const unsigned long COLS_B           = 10000;
const unsigned long ROWS_B           = COLS_A;
const unsigned long INNER_DIM        = COLS_A;
const unsigned long ROWS_C           = ROWS_A;
const unsigned long COLS_C           = COLS_B;
```

Given the dimensions of the matrices, we need to compute the padded dimensions
as well. This is to satisfy the quantization problem discussed above. We use
the `RND_UP_MULT` macro to make this easier. The desired multiples are four
times the corresponding `MNK` values. The factor of four is necessary
because—as we'll see below—every thread block is responsible for a square of 16
tiles in the output.

```cpp
const unsigned long ROWS_A_PADDED    = RND_UP_MULT(ROWS_A, WMMA_M * 4);
const unsigned long COLS_A_PADDED    = RND_UP_MULT(COLS_A, WMMA_K * 4);
const unsigned long COLS_B_PADDED    = RND_UP_MULT(COLS_B, WMMA_N * 4);
const unsigned long ROWS_B_PADDED    = COLS_A_PADDED;
const unsigned long INNER_DIM_PADDED = COLS_A_PADDED;
const unsigned long ROWS_C_PADDED    = ROWS_A_PADDED;
const unsigned long COLS_C_PADDED    = COLS_B_PADDED;
```

Every kernel must define its block and grid dimensions. First, it's important
to understand that the tensor cores operate at the warp level. That is, all 32
threads in a warp must collaborate to execute any of the functions in the
`wmma` namespace. An easy way to keep track of that is to make the `x`
dimension of the block size equal to the warp size. Thread IDs count along the
`x` dimension first, so this all works out nicely.

Here's an area where I'm slightly confused. At first, I set the `y` and `z`
dimensions of the block size to 1, since that seemed like a reasonable guess.
When I ran the program through the profiler, it complained about low
[occupancy][occupancy].  I don't fully understand occupancy, but it seems like
it's more or less a measure of how well the blocks fit in the streaming
multiprocessors (SMs). If occupancy is low, then the kernel isn't using all of
the SM's resources. I think of this kind of like packing a car. If your bags
and boxes are too big, then you can't fill up all the available volume, but if
they're small then you can pack more efficiently by fitting them together.

[occupancy]: https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm

Anyway, I played with the `y` and `z` dimensions for a while to increase the
occupancy, and I landed on 4. It's almost completely arbitrary and probably not
optimal, but it works alright. The net results are that occupancy is increased,
which *generally* increases speed, and each block is now responsible for 16
outputs instead of just one.

We also have to set up the grid dimensions, which need to work out so that
there are exactly enough blocks to cover the entire output. This is one of many
cases in this kind of programming where careful arithmetic is essential.

TODO a diagram would be helpful here; how do the blocks map to the output matrix?

```cpp
const unsigned int  WARP_SIZE        = 32;
const dim3          NUM_THREADS        (WARP_SIZE,4,4);
const dim3          NUM_BLOCKS         (ROWS_C_PADDED / (WMMA_M * NUM_THREADS.y),
                                        COLS_C_PADDED / (WMMA_N * NUM_THREADS.z),
                                        1);
```

When we run the kernel and measure its execution time, we'll want to estimate
how many operations it ran per second. To do that, we need a credible estimate
for the numerator—the number of operations. We're going to conflate
multiplication and addition operations here just to keep things simple, and
we're ignoring the existence of fused multiply-add instructions, too. With
those simplifications, we come to the expression below. First, we have to
multiply every element of $\mathbf{A}$ by a scalar, so that's $M \times K$
operations. For every element in $\mathbf{C}$, we know that we have to multiply
$K$ pairs of elements. We also have to add them all together, which adds
another $K - 1$ operations *per element of* $\mathbf{C}$. The scalar
multiplication by $\beta$ and the final summation each add another $M \times N$
operations, giving a total of

$$MK + (K + K - 1)MN + 2MN$$

I call this value the total number of "effective operations" or "eOPS". The "e"
is there to remind the reader that this value isn't really a perfect
representation of what the kernel is actually doing. It's more like a measure
of how many operations you *would* have to do if you were doing GEMM the naive
way, such as in the CPU verification routine.

```cpp
const unsigned long long TOTAL_EOPS = (ROWS_A * INNER_DIM)
                                      + (INNER_DIM + INNER_DIM - 1) * (ROWS_C * COLS_C)
                                      + 2 * (ROWS_C * COLS_C);
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
signature first. `A`, `B`, and `C` are pointers to arrays, and `r_A`, `inner`,
and `c_B` represent their dimensions. Three dimensions are enough here because
`A` and `B` must have equal numbers of columns and rows, respectively, and `C`
must have `r_A` rows and `c_B` columns. `alpha` and `beta` are the scalar
coefficients.

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

Next we set a convenient namespace and assign some useful variables. Most of
the stuff we touch in this kernel is in the `nvcuda::wmma` namespace, so
changing the namespace in just this scope improves readability. `warp_row` and
`warp_col` describe what output tile this warp is responsible for. Recall that
tensor core operations need an entire warp to cooperate, and we defined our
block and grid dimensions to account for that. These two variables tell us
where the top-left corner element is in `C`. `tile_run` tells us how many times
we need to move along the inner dimension to accumulate all of the
contributions from the inputs.

TODO diagram showing how `warp_{row,col}` map to the matrices

If this is confusing, it may help to remember that while this implementation
does use the tensor cores, it's still more or less a naive CUDA matmul. We're
still doing dot products over the rows and columns of the inputs. The big
difference is that the "elements" of the matmul are now these multi-element
tiles. If you're familiar with element-wise tiling, this is basically the same
thing but with different function calls.

```cpp
    using namespace nvcuda::wmma;
    int warp_row = (blockIdx.x * blockDim.y + threadIdx.y) * WMMA_M;
    int warp_col = (blockIdx.y * blockDim.z + threadIdx.z) * WMMA_N;
    int tile_run = inner / WMMA_K;
```

The tensor core API requires that we load data into `fragment`s, which are
opaque structures that contain some matrix elements to be used in an operation.
We have to explicitly declare which kind of matrix a fragment contains in the
matmul operation (`matrix_a`, `matrix_b`, or `accumulator`), as well as how big
the fragment is, what type the elements are, and how to index the source
matrices (`row_major` or `col_major`). That last bit gives us a convenient way
to accomplish transposition. If the corresponding input matrix is supposed to
be transposed, we declare the fragment as `col_major`. The `accumulator`
fragment is not affected by transposition.

We also define an index expression that accounts for the transposition. Note
that the `x` and `y` arguments to `IDX` are flipped, and also we refer to an
undeclared variable `i`. When we use this macro in a loop below, the `i`
variable will contain the inner dimension position.

At the top of this block, we also have to optionally change the fragment type
when using `tf32`. The reason is that `tf32` seems to act more or less like a
passthrough type for `float`, even though the bits are different. According to
the [Nvidia docs][tf32-docs], you have to pass this alternate type to the
fragments in order to get the benefits of `tf32` on the tensor cores. I'm not
certain I'm doing this right, since the docs also mention `__float_to_tf32`,
which I can't find anywhere else online, and which I haven't used in this
kernel. In all other cases, the actual input type is sufficient.

[tf32-docs]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alternate-floating-point

```cpp
    #if defined(TF32_IN_F32_OUT)
        #define FRAGMENT_TYPE precision::tf32
    #else
        #define FRAGMENT_TYPE INPUT_ELEMENT
    #endif
    #if TRANSPOSE_A
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, FRAGMENT_TYPE, col_major> A_frag;
        #define IDX_A IDX(i * WMMA_K, warp_row, inner)
    #else
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, FRAGMENT_TYPE, row_major> A_frag;
        #define IDX_A IDX(warp_row, i * WMMA_K, inner)
    #endif
    #if TRANSPOSE_B
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, FRAGMENT_TYPE, col_major> B_frag;
        #define IDX_B IDX(warp_col, i * WMMA_K, c_B)
    #else
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, FRAGMENT_TYPE, row_major> B_frag;
        #define IDX_B IDX(i * WMMA_K, warp_col, c_B)
    #endif
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, OUTPUT_ELEMENT> C_frag;
```

We'll begin in earnest by loading the output tile into the `accumulator`
fragment and scaling it by `beta`. `load_matrix_sync` takes the target
fragment, a pointer to the top-left element, the leading dimension (columns of
`C` in this case), and the storage layout. It coordinates all of the threads in
the warp to do the load from global memory `sync`hronously. Then, we iterate
over each element and scale by `beta`.[^scaling] We do this before the matmul
because we're going to accumulate partial results in `C_frag`, which would make
it impossible to scale just the original values by `beta` later on.

[^scaling]: It's just occuring to me now that this might not make sense. Each
thread in the warp is going to run this `for` loop, so it seems like the
scaling should be done multiple times per element, which isn't what we want.
The kernel's output matches against `h_gemm`, so maybe the compiler is smart
enough to see what we're doing and parallelize it across the warp. Or maybe I'm
wrong about this being a problem at all.

```cpp
    load_matrix_sync(C_frag, C + IDX(warp_row, warp_col, c_B), c_B, mem_row_major);
    for (int i = 0; i < C_frag.num_elements; i++) {
        #if defined(F16_IN_F32_OUT)
            C_frag.x[i] *= __half2float(beta);
        #elif defined(BF16_IN_F32_OUT)
            C_frag.x[i] *= __bfloat162float(beta);
        #else
            C_frag.x[i] *= beta;
        #endif
    }
```

Just like an ordinary matrix multiplication, we walk along the inner dimension,
multiplying and accumulating tiles as we go. We also scale the elements in the
tile from `A` by `alpha`. `mma_sync` runs the multiply-add operation over the
last three arguments and accumulates the result in the first argument. It also
supports this "in-place" operation where the first and last arguments are the
same. At the end of the outer loop, `C_frag` contains the correct GEMM outputs
for this warp's tile.

TODO: like previous diagram, but emphasize walking along K dimension

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

Finally, we call another warp-level function to store the values of `C_frag`
into the correct location in `C`, which is ultimately a store operation into
global memory.

```cpp
    store_matrix_sync(C + IDX(warp_row, warp_col, c_B), C_frag, c_B, mem_row_major);
}
```

# Main Function

Now we come to the entry point of the program. Everything from here on is more
or less just a test harness for the kernel we saw above.

```cpp
int main() {
```

## Printing Test Information

For convenience in testing, we begin by printing out some decently formatted
information about the GEMM operation we're about to run. We also check that the
dimensions of the blocks and grids match the number of elements in the output
matrix. It's easy to get these wrong because reasoning about all the different
loops and dimensions is often difficult. This runtime check catches a decent
number of errors in this category.

To allocate memory for our three matrices, we need to know their sizes in
bytes. We'll compute those values now by finding the number of elements and
then multiplying that value by the number of bytes per element. We also print
the dimensions and sizes of the matrices.

It's useful to know the [arithmetic intensity][arith-intens] of the
computation, which is the ratio of the total number of math operations to the
total number of bytes that need to move during the computation. This figure
gives some insight into whether or not the computation will be limited by the
compute performance or memory bandwidth of the target platform.

[arith-intens]: https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity

The code in this area is somewhat verbose and not at all interesting, so I've
hidden it.

<details>
<summary>Click to show/hide test information code</summary>

```cpp
    #if defined(F16_IN_F32_OUT) || defined(F16_IN_OUT)
        printf("ALPHA        : %6f\n", __half2float(ALPHA));
        printf("BETA         : %6f\n\n", __half2float(BETA));
    #elif defined(BF16_IN_F32_OUT)
        printf("ALPHA        : %6f\n", __bfloat162float(ALPHA));
        printf("BETA         : %6f\n\n", __bfloat162float(BETA));
    #elif defined(U8_IN_I32_OUT) || defined(I8_IN_I32_OUT)
        printf("ALPHA        : %6d\n", ALPHA);
        printf("BETA         : %6d\n\n", BETA);
    #else
        printf("ALPHA        : %6f\n", ALPHA);
        printf("BETA         : %6f\n\n", BETA);
    #endif

    printf("TRANSPOSE_A  : %s\n", TRANSPOSE_A ? "true" : "false");
    printf("TRANSPOSE_B  : %s\n\n", TRANSPOSE_B ? "true" : "false");

    printf("WMMA_M       : %3lu\n", WMMA_M);
    printf("WMMA_N       : %3lu\n", WMMA_N);
    printf("WMMA_K       : %3lu\n\n", WMMA_K);

    printf("NUM_BLOCKS   : (%3d, %3d, %3d) = %5d\n",
           NUM_BLOCKS.x,
           NUM_BLOCKS.y,
           NUM_BLOCKS.z,
           NUM_BLOCKS.x * NUM_BLOCKS.y * NUM_BLOCKS.z
    );
    printf("NUM_THREADS  : (%3d, %3d, %3d) = %5d\n\n",
           NUM_THREADS.x,
           NUM_THREADS.y,
           NUM_THREADS.z,
           NUM_THREADS.x * NUM_THREADS.y * NUM_THREADS.z
    );

    #if defined(F16_IN_OUT)
        printf("Type:        : F16_IN_OUT\n\n");
    #elif defined(F16_IN_F32_OUT)
        printf("Type         : F16_IN_F32_OUT\n\n");
    #elif defined(U8_IN_I32_OUT)
        printf("Type         : U8_IN_I32_OUT\n\n");
    #elif defined(I8_IN_I32_OUT)
        printf("Type         : I8_IN_I32_OUT\n\n");
    #elif defined(BF16_IN_F32_OUT)
        printf("Type         : BF16_IN_F32_OUT\n\n");
    #elif defined(TF32_IN_F32_OUT)
        printf("Type         : TF32_IN_F32_OUT\n\n");
    #elif defined(F64_IN_OUT)
        printf("Type         : F64_IN_OUT\n\n");
    #elif defined(U4_IN_I32_OUT)
        printf("Type         : U4_IN_I32_OUT\n\n");
    #elif defined(S4_IN_I32_OUT)
        printf("Type         : S4_IN_I32_OUT\n\n");
    #elif defined(B1_IN_I32_OUT)
        printf("Type         : B1_IN_I32_OUT\n\n");
    #endif

    size_t SIZE_A = ROWS_A * COLS_A * sizeof(INPUT_ELEMENT);
    printf(
        "A original   : %5lu × %5lu, %lu B => %9lu B\n",
        ROWS_A,
        COLS_A,
        sizeof(INPUT_ELEMENT),
        SIZE_A
    );
    size_t SIZE_A_PADDED = ROWS_A_PADDED * COLS_A_PADDED * sizeof(INPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu, %lu B => %9lu B\n",
        ROWS_A_PADDED,
        COLS_A_PADDED,
        sizeof(INPUT_ELEMENT),
        SIZE_A_PADDED
    );
    size_t SIZE_B = ROWS_B * COLS_B * sizeof(INPUT_ELEMENT);
    printf(
        "B original   : %5lu × %5lu, %lu B => %9lu B\n",
        ROWS_B,
        COLS_B,
        sizeof(INPUT_ELEMENT),
        SIZE_B
    );
    size_t SIZE_B_PADDED = ROWS_B_PADDED * COLS_B_PADDED * sizeof(INPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu, %lu B => %9lu B\n",
        ROWS_B_PADDED,
        COLS_B_PADDED,
        sizeof(INPUT_ELEMENT),
        SIZE_B_PADDED
    );
    size_t SIZE_C = ROWS_C * COLS_C * sizeof(OUTPUT_ELEMENT);
    printf(
        "C original   : %5lu × %5lu, %lu B => %9lu B\n",
        ROWS_C,
        COLS_C,
        sizeof(OUTPUT_ELEMENT),
        SIZE_C
    );
    size_t SIZE_C_PADDED = ROWS_C_PADDED * COLS_C_PADDED * sizeof(OUTPUT_ELEMENT);
    printf(
        "    padded   : %5lu × %5lu, %lu B => %9lu B\n\n",
        ROWS_C_PADDED,
        COLS_C_PADDED,
        sizeof(OUTPUT_ELEMENT),
        SIZE_C_PADDED
    );
    size_t TOTAL_SIZE = SIZE_A + SIZE_B + SIZE_C;
    printf("Total eOPS   : %13llu\n", TOTAL_EOPS);
    printf("Input size   : %13lu B\n", SIZE_A + SIZE_B);
    printf("Output size  : %13lu B\n", SIZE_C);
    printf(
        "eOPS per byte: %13f\n\n",
        static_cast<double>(TOTAL_EOPS) / static_cast<double>(TOTAL_SIZE)
    );
    assert(
        NUM_BLOCKS.x * NUM_THREADS.y * WMMA_N
        * NUM_BLOCKS.y * NUM_THREADS.z * WMMA_M
        == ROWS_C_PADDED * COLS_C_PADDED
    );
```

</details>

Now we'll set the random seed to a fixed value so we always get the same
element values for a given configuration. This eliminates a source of random
variation in performance and makes debugging less frustrating.

```cpp
    srand(0);
```

## Host Allocation and Initialization

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
*next* rows and exceeding the boundaries of the matrices on the last rows.
Instead, we need to allocate padded versions of the matrices with dimensions
that *are* a multiple of the tensor core dimensions and fill the unneeded
values with zeros so they don't affect the computation. Then, after the kernel
is done running, we'll need to extract the values we want from the output to
get the true result.[^kernel-padding]

TODO diagram demonstrating overrun problem and padding solution

[^kernel-padding]: It would be better to do this padding in the kernels because
the time spent padding would be parallelized. I haven't bothered to do that
here, and I think it requires some nontrivial changes. I'm not sure how to
handle the padded tiles if the given device array doesn't have enough capacity
for them.

```cpp
    INPUT_ELEMENT* h_A_padded = (INPUT_ELEMENT*)malloc(SIZE_A_PADDED);
    INPUT_ELEMENT* h_B_padded = (INPUT_ELEMENT*)malloc(SIZE_B_PADDED);
    OUTPUT_ELEMENT* h_C_padded = (OUTPUT_ELEMENT*)malloc(SIZE_C_PADDED);
```

We want the padded matrices to have identical values to the unpadded ones where
they're defined and zeros elsewhere. The logic below accomplishes that for
`h_A_padded`, and I've hidden the rest because it's the same thing with
different pointers and iteration boundaries. We also time this operation and
print the duration. For size 10,000 square matrices and `I8_IN_I32_OUT`, this
takes about 115 ms on the Jetson's CPU.

Rather than touch every element individually, this logic uses `memcpy` to grab
an entire row of the unpadded matrix and copy it to the padded version. Then,
it fills in zeros in the remaining columns. For any extra rows, separate logic
fills one row with zeros and then `memcpy`s those as needed.

```cpp
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    for (int r = 0; r < ROWS_A; r++) {
        memcpy(
            h_A_padded + IDX(r, 0, COLS_A_PADDED),
            h_A + IDX(r, 0, COLS_A),
            COLS_A * sizeof(INPUT_ELEMENT)
        );
        for (int c = COLS_A; c < COLS_A_PADDED; c++) {
            h_A_padded[IDX(r, c, COLS_A_PADDED)] = static_cast<INPUT_ELEMENT>(0);
        }
    }
    for (int c = 0; c < COLS_A_PADDED; c++) {
        h_A_padded[IDX(ROWS_A, c, COLS_A_PADDED)] = 0;
    }
    for (int r = ROWS_A + 1; r < ROWS_A_PADDED; r++) {
        memcpy(
            h_A_padded + IDX(r, 0, COLS_A_PADDED),
            h_A_padded + IDX(ROWS_A, 0, COLS_A_PADDED),
            COLS_A_PADDED * sizeof(INPUT_ELEMENT)
        );
    }
```

<details>
<summary>Click to show/hide padding logic for other matrices</summary>

```cpp
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
    for (int c = 0; c < COLS_B_PADDED; c++) {
        h_B_padded[IDX(ROWS_B, c, COLS_B_PADDED)] = 0;
    }
    for (int r = ROWS_B + 1; r < ROWS_B_PADDED; r++) {
        memcpy(
            h_B_padded + IDX(r, 0, COLS_B_PADDED),
            h_B_padded + IDX(ROWS_B, 0, COLS_B_PADDED),
            COLS_B_PADDED * sizeof(INPUT_ELEMENT)
        );
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
    for (int c = 0; c < COLS_C_PADDED; c++) {
        h_C_padded[IDX(ROWS_C, c, COLS_C_PADDED)] = 0;
    }
    for (int r = ROWS_C + 1; r < ROWS_C_PADDED; r++) {
        memcpy(
            h_C_padded + IDX(r, 0, COLS_C_PADDED),
            h_C_padded + IDX(ROWS_C, 0, COLS_C_PADDED),
            COLS_C_PADDED * sizeof(INPUT_ELEMENT)
        );
    }
```

</details>

```cpp
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    unsigned long pad_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    printf("Padding      : %13lu μs\n\n", pad_elapsed);
```

If we're going to do CPU verification later, then we need an original copy of
the `C` matrix so we can run GEMM a second time. The GEMM operation modifies
`C` in the general case, so running the kernel changes the values in `h_C`
after copying them back from the GPU.  We'll save a copy of the original values
in `h_C_orig`. We don't actually need to do this unless `DO_CPU_VERIFY` is
`true`, but adding the extra logic to handle conditionally allocating and
freeing the memory isn't worth the benefit in my opinion, so we just do it
every time.

```cpp
    OUTPUT_ELEMENT* h_C_orig = (OUTPUT_ELEMENT*)malloc(SIZE_C);
    memcpy(h_C_orig, h_C, SIZE_C);
```

If `DEBUG_OUTPUT` is `true`, then we'll print the initial contents of `A`, `B`,
and `C`. The `printMat` function is generic over the element type—just like
`initMatrix` above—and prints the matrix elements in a format that can be
easily copied into a Python REPL.

```cpp
    if (DEBUG_OUTPUT) {
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
    cudaInitDevice(0, cudaDeviceScheduleAuto, 0);
```

## Kernel Launch and Speed Measurement

We're finally done setting up. We can launch the kernel and measure the time it
takes to execute. We pass `NUM_BLOCKS` and `NUM_THREADS` as the launch
parameters, and we pass device pointers and other constants as arguments to the
kernel. The call to `cudaDeviceSynchronize` guarantees that the GPU will finish
its most recent kernel before advancing to the next line. Normally, CPU
execution continues after a kernel launch—which is the point of having a GPU at
all—and that's usually the desired behavior. In this case, we *want* to block
on the kernel's execution because we're trying to measure how long it takes to
run.

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

Now we'll get the number of microseconds between the start and end of the
kernel's runtime and then report its effective speed.

```cpp
    long d_elapsed = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double eops = static_cast<double>(TOTAL_EOPS) / (static_cast<double>(d_elapsed) / 1e6);
    printf("Device       : %13lu μs (%11f eTOPS)\n", d_elapsed, eops / 1e12);
```

To complete the operation---and to optionally check its correctness---we need
to copy the result back to the CPU and undo the padding operation we did
earlier. This logic is a lot simpler than the initial padding. We'll also
measure and report its duration a little further down. For the case I mentioned
above, the total execution time is about 50 ms on the CPU.

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

If the corresponding constants are enabled, then we'll print out the matrix
contents and/or run the CPU version of GEMM to check our results from the GPU.
For comparison, it's also fun to measure the runtime of the naive CPU
implementation to see how much faster the GPU is.

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
            "Host         : %13lu μs (%11f eTOPS)",
            h_elapsed,
            (static_cast<double>(TOTAL_EOPS) / (static_cast<double>(h_elapsed) / 1e6)) / 1e12
        );
        if (DEBUG_OUTPUT) {
            printf("CPU result:\n");
            printMat<OUTPUT_ELEMENT>(h_C_orig, ROWS_C, COLS_C);
        }
        if (verify(h_C_orig, h_C)) {
            printf(" CORRECT\n\n");
        } else {
            printf(" WRONG\n\n");
        }
    } else {
        printf("Host         :       skipped\n\n");
    }
    printf("Extraction   : %13lu μs\n", extract_elapsed);
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

## Remarks on Optimization

I spent a lot of late nights learning about optimization techniques for this
kernel and trying to implement them. (It was not healthy.) I became reluctant
friends with `ncu` and Nsight Compute, the profiling tools for CUDA kernels.
All I can say for sure after all that work is that I have only begun to
understand a small fraction of what's out there. But, I'm told this is a good
sign because it means I've left the safety of the [Dunning-Kruger
effect][dk-effect].

[dk-effect]: https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect

As written, `d_gemm` does not make optimal use of the resources on the Jetson.
In particular, it seems to be memory-bound, which makes sense to me because the
kernel is constantly loading tiles from global memory, and adjacent warps are
loading *the same tiles* from `A` and `B`. I tried doing a "tiling of tiles"
with shared memory, but aside from a lot of headaches with indexing, I didn't
get too far. I never wrote a correct implementation, but once my best attempt
started getting close, it became clear that it wasn't going to be faster.

There is definitely room for improvement, but it's notable that cuBLAS runs
SGEMM (single-precision GEMM) at about 1.7 eTOPS, while this implementation
gets about 0.8 eTOPS (~47%) with `TF32_IN_F32_OUT`. I don't know if that's a
fair comparison, but it's probably somewhat reasonable. I'm ~~happy with~~
ready to accept 2x headroom for this project, since it's nominally about
learning (like all my projects).

Notably, transposition seems to make a big difference in performance. When
`TRANSPOSE_B` is `true`, the performance for `I8_IN_I32_OUT` increases by about
3x. As with other implementations, I guess this is due to improved coalescing /
cache performance when reading `B` as rows instead of columns. I definitely
don't understand how to make optimal use of this yet.

If I do come back to this in the future to apply optimizations, I'll probably
start by trying to understand what cuBLAS does that's so much faster. It seems
more likely that when I get back around to playing with GPUs again, I'll want
to use my basic general CUDA knowledge to work on a different kernel where
custom code is valuable, such as fusion of otherwise discrete kernels to save
on memory traffic.

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
            #if defined(F16_IN_F32_OUT)
                C[IDX(row, col, c_B)] *= __half2float(beta);
            #elif defined(BF16_IN_F32_OUT)
                C[IDX(row, col, c_B)] *= __bfloat162float(beta);
            #else
                C[IDX(row, col, c_B)] *= beta;
            #endif
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
                #if defined(F16_IN_F32_OUT)
                    C[IDX(row, col, c_B)] += __half2float(alpha * a * b);
                #elif defined(BF16_IN_F32_OUT)
                    C[IDX(row, col, c_B)] += __bfloat162float(alpha * a * b);
                #else
                    C[IDX(row, col, c_B)] += alpha * a * b;
                #endif
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
specifications][conv-spec].

[flop-error]: https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems
[conv-spec]: https://en.cppreference.com/w/cpp/io/c/fprintf

```cpp
bool verify(OUTPUT_ELEMENT* h_solution, OUTPUT_ELEMENT* h_C) {
    for (int idx = 0; idx < ROWS_C * COLS_C; idx++) {
        #if defined(F16_IN_OUT)
            if (__hgt(__habs(h_C[idx] - h_solution[idx]), VERIFY_TOLERANCE)) {
        #else
            if (abs(h_C[idx] - h_solution[idx]) > VERIFY_TOLERANCE) {
        #endif
            printf(
                #if defined(U8_IN_I32_OUT) || defined(I8_IN_I32_OUT)
                    "Found output mismatch at (%lu,%lu): device returned %d but solution is %d\n",
                #else
                    "Found output mismatch at (%lu,%lu): device returned %f but solution is %f\n",
                #endif
                idx / COLS_C,
                idx % COLS_C,
                #if defined(F16_IN_OUT)
                    __half2float(h_C[idx]),
                    __half2float(h_solution[idx])
                #elif defined(BF16_IN_F32_OUT)
                    __bfloat162float(h_C[idx]),
                    __bfloat162float(h_solution[idx])
                #else
                    h_C[idx],
                    h_solution[idx]
                #endif
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
be surrounded by `[]` and the elements and rows must be separated by a comma.
We loop over the rows and colums, emitting characters and elements as necessary
to satisfy these requirements. As with `verify` above, different `printf` calls
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
            #elif defined(U8_IN_I32_OUT)
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
