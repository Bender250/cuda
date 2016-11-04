// CUDA-C includes
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

const unsigned int THREAD_COUNT = 64;
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

__global__ void avg_div(float *array, const unsigned int delimiter) {
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


    dim3 threadsPerBlock(THREAD_COUNT,1);
    dim3 numBlocks(questions/(THREAD_COUNT*THREAD_SIZE), students/Y_TILE_SIZE);
    parallel_col_sum<<<numBlocks, threadsPerBlock>>>(results, avg_stud, avg_que, questions);
    avg_div<<<students/THREAD_COUNT, THREAD_COUNT>>>(avg_stud, questions);
    avg_div<<<questions/THREAD_COUNT, THREAD_COUNT>>>(avg_que, students);

    // if input is students << questions, switch to rowsum without caching
}

