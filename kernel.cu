// CUDA-C includes
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

//#include <stdio_ext.h>

//const unsigned int LOG_BLOCK_SIZE = 5;
const unsigned int THREAD_COUNT = 64;//1 << LOG_BLOCK_SIZE;
const unsigned int THREAD_SIZE = 8;
const unsigned int Y_TILE_SIZE = 64;

// hacking speed - uint16_t?

__global__ void parallel_col_sum(const int *in, float *avg_stud, float *avg_que,
                                 const unsigned int questions) {
    unsigned int tx = threadIdx.x;
    unsigned int blocks_row = blockIdx.y*Y_TILE_SIZE;

    __shared__ int tile[THREAD_COUNT*THREAD_SIZE];
    __shared__ int tile_accum;

    int col_res_accum[THREAD_COUNT] = {0}; // array per thread, in registers

    unsigned int col_offset = blockIdx.x*(THREAD_COUNT*THREAD_SIZE);

    // Iterate over ROWS
    for (unsigned int row = blocks_row; row < blocks_row + Y_TILE_SIZE; ++row) {

        // copy input to shared mem by BLOCK_SIZE blocks (done in parallel in blocks of threads)
        // for BLOCK_SIZE*THREAD_SIZE
        // unroll
        for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
            tile[i*THREAD_COUNT + tx] = in[row*questions + col_offset + i*THREAD_COUNT + tx];
        }

        if (tx == 0)
            tile_accum = 0;
        __syncthreads(); // if ( BLOCK_SIZE == 32 ) not necessary

        int row_accum_thread = 0; // per thread
        for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
            row_accum_thread += tile[i*THREAD_COUNT + tx];
            col_res_accum[i] += tile[i*THREAD_COUNT + tx];
        }
        __syncthreads(); // if ( BLOCK_SIZE == 32 ) not necessary - speeedup


        atomicAdd(&tile_accum, row_accum_thread);
        __syncthreads();
        if (tx == 0)
            atomicAdd((avg_stud + row), tile_accum);
        __syncthreads();
    }

    // store que results
    for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
        atomicAdd((avg_que + col_offset + i*THREAD_COUNT + tx),
                  col_res_accum[i]);
    }
}

/* <200 MEvals - just 1/36 WARP runing
__global__ void parallel_col_sum(const int *in, float *avg_stud, float *avg_que,
                                 const unsigned int students, const unsigned int questions) {
    unsigned int tx = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x*THREAD_SIZE + tx;

    __shared__ int tile[BLOCK_SIZE*THREAD_SIZE];
    __shared__ int row_accum;

    int col_res_accum[BLOCK_SIZE] = {0};

    for (unsigned int row = 0; row < students; ++row) {

        // copy input to shared mem by BLOCK_SIZE blocks (done in parallel in blocks of threads)
        // for BLOCK_SIZE*THREAD_SIZE
        // unroll
        unsigned int offset = row * questions;
        for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
            tile[i*THREAD_SIZE + tx] = in[offset + i*THREAD_SIZE + index];
        }

        row_accum = 0; // fight between threads, but set it to 0!
        __syncthreads(); // if ( BLOCK_SIZE == 32 ) not necessary

        int row_accum_register = 0;
        for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
            row_accum_register += tile[tx*THREAD_SIZE + i];
            col_res_accum[i] += tile[tx*THREAD_SIZE + i];
        }
        //__sync?

        atomicAdd(&row_accum, row_accum_register);
        __syncthreads();
        atomicAdd((avg_stud + row), float(row_accum));
        __syncthreads();
    }

    // store student results
    for (unsigned int i = 0; i < THREAD_SIZE; ++i) {
        avg_que[i*THREAD_SIZE + index] = float(col_res_accum[i*THREAD_SIZE + tx])/float(students);
    }
}
*/

__global__ void tiled_sum(const int *in, float *avg_stud, float *avg_que, const int students, const int questions) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_que = 0;

    for (int j = 0; j < students; ++j) {
        __shared__ int tile[THREAD_COUNT + THREAD_COUNT];

        tile[threadIdx.x] = in[questions*j + i];
        __syncthreads();

        tmp_que += tile[threadIdx.x];

        // 1420 ME for 32 block_size, 1650 ME for 16
        if (threadIdx.x == 0) {
            int tmp_stud = 0;
            for (unsigned int k = 0; k < THREAD_COUNT; ++k) {
                tmp_stud += tile[k];
            }
            atomicAdd((avg_stud + j), float(tmp_stud));
        }
        __syncthreads();

        /*
        // 900 MEvals
        // unroll
        int t_partial_sum = tile[threadIdx.x];

        for (int offset = BLOCK_SIZE; offset > 32; offset >>= 1) {
            if (threadIdx.x < offset) {
                tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + 32];
            tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + 16];
            tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + 8];
            tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + 4];
            tile[threadIdx.x] = t_partial_sum = t_partial_sum + tile[threadIdx.x + 2];
                                t_partial_sum = t_partial_sum + tile[threadIdx.x + 1];
        }

        atomicAdd((avg_stud + j), float(t_partial_sum));
        __syncthreads();*/
    }

    avg_que[i] = float(tmp_que) / float(students);
}

__global__ void avg_div(float *array, const unsigned int delimiter, const unsigned int pos) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    array[i] = array[i] / float(delimiter);
}

__global__ void col_sum(const int *in, float *avg_que, const int students, const int questions) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_que = 0;

    for (int j = 0; j < students; ++j) {
        tmp_que += in[questions*j + i];
    }

    avg_que[i] = float(tmp_que) / float(students);
}

__global__ void row_sum(const int *in, float *avg_stud, const int questions) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_stud = 0;

    for (int j = 0; j < questions; ++j) {
        tmp_stud += in[questions*i + j];
    }

    avg_stud[i] = float(tmp_stud) / float(questions);
}

void solveGPU(const int *results, float *avg_stud, float *avg_que,
              const unsigned int students, const unsigned int questions) {
    cudaMemset(avg_stud, 0.0, students*4);
    cudaMemset(avg_que, 0.0, questions*4);

    //col_sum<<<questions/BLOCK_SIZE, BLOCK_SIZE>>>(results, avg_que, students, questions);
    //row_sum<<<students/BLOCK_SIZE, BLOCK_SIZE>>>(results, avg_stud, questions);

    // not working yet

    //tiled_sum<<<questions/BLOCK_SIZE, BLOCK_SIZE>>>(results, avg_stud, avg_que, students, questions);
    //parallel_col_sum<<<questions/(BLOCK_SIZE*THREAD_SIZE), BLOCK_SIZE>>>(results, avg_stud, avg_que, students, questions);
    //avg_div<<<students/BLOCK_SIZE, BLOCK_SIZE>>>(avg_stud, questions);


    dim3 threadsPerBlock(THREAD_COUNT,1);
    dim3 numBlocks(questions/(THREAD_COUNT*THREAD_SIZE), students/Y_TILE_SIZE);
    parallel_col_sum<<<numBlocks, threadsPerBlock>>>(results, avg_stud, avg_que, questions);
    avg_div<<<students/THREAD_COUNT, THREAD_COUNT>>>(avg_stud, questions, (students / 4));
    avg_div<<<questions/THREAD_COUNT, THREAD_COUNT>>>(avg_que, students, (questions / 4));

    // if input is students << questions, switch to rpwsum without caching
}

