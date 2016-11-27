// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

const u_int16_t THREAD_COUNT = 64;
const u_int16_t THREAD_SIZE = 8;
const u_int16_t Y_TILE_SIZE = 64;

/**
 * suitable only for matrices of X % (64*8) = 0; Y % 64 = 0
 */
template <u_int16_t threads, u_int16_t cols_per_thread,
          u_int16_t rows_per_thread>
__global__ void parallel_col_sum(const int *in, float *avg_row, float *avg_col,
                                 const u_int16_t cols) {
  u_int16_t tx = u_int16_t(threadIdx.x);
  u_int16_t blocks_row = u_int16_t(blockIdx.y) * rows_per_thread;

  __shared__ u_int16_t tile[threads * cols_per_thread];
  __shared__ int tile_accum;

  u_int16_t col_res_accum[threads] = {0}; // array per thread, in registers

  u_int16_t col_offset = u_int16_t(blockIdx.x) * (threads * cols_per_thread);

  // Iterate over ROWS
  for (u_int16_t row = blocks_row; row < blocks_row + rows_per_thread; ++row) {

// copy input to shared mem by threads blocks (done in parallel in blocks
// of threads)
// for threads*cols_per_thread
#pragma unroll
    for (u_int16_t i = 0; i < cols_per_thread; ++i) {
      tile[i * threads + tx] =
          u_int16_t(in[row * cols + col_offset + i * threads + tx]);
    }

    // if (tx == 0)
    tile_accum = 0;
    __syncthreads(); // if ( threads == 32 ) not necessary

    u_int16_t row_accum_thread = 0; // per thread
#pragma unroll
    for (u_int16_t i = 0; i < cols_per_thread; ++i) {
      row_accum_thread += tile[i * threads + tx];
      col_res_accum[i] += tile[i * threads + tx];
    }
    //__syncthreads(); // if ( threads == 32 ) not necessary - speeedup

    atomicAdd(&tile_accum, row_accum_thread);
    __syncthreads();
    if (tx == 0)
      atomicAdd((avg_row + row), tile_accum);
    //__syncthreads();
  }

  // store que results
  for (u_int16_t i = 0; i < cols_per_thread; ++i) {
    atomicAdd((avg_col + col_offset + i * threads + tx), col_res_accum[i]);
  }
}

__global__ void avg_div(float *array, const u_int16_t delimiter) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  array[i] /= float(delimiter);
}

__global__ void col_sum(const int *in, float *avg_que, const int students,
                        const int questions) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp_que = 0;

  for (int j = 0; j < students; ++j) {
    tmp_que += in[questions * j + i];
  }

  avg_que[i] = float(tmp_que) / float(students);
}

__global__ void row_sum(const int *in, float *avg_stud, const int questions) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp_stud = 0;

  for (int j = 0; j < questions; ++j) {
    tmp_stud += in[questions * i + j];
  }

  avg_stud[i] = float(tmp_stud) / float(questions);
}

void solveGPU(const int *results, float *avg_stud, float *avg_que,
              const unsigned int students, const unsigned int questions) {
  cudaMemset(avg_stud, 0.0, students * 4);
  cudaMemset(avg_que, 0.0, questions * 4);

  dim3 threadsPerBlock(THREAD_COUNT, 1);
  dim3 numBlocks(questions / (THREAD_COUNT * THREAD_SIZE),
                 students / Y_TILE_SIZE);
  if ((questions % (64 * 8) == 0) && (students % 64 == 0)) {
    parallel_col_sum<THREAD_COUNT, THREAD_SIZE,
                     Y_TILE_SIZE><<<numBlocks, threadsPerBlock>>>(
        results, avg_stud, avg_que, u_int16_t(questions));
    avg_div<<<students / THREAD_COUNT, THREAD_COUNT>>>(avg_stud, u_int16_t(questions));
    avg_div<<<questions / THREAD_COUNT, THREAD_COUNT>>>(avg_que, u_int16_t(students));
  } else {
    col_sum<<<questions / THREAD_COUNT, THREAD_COUNT>>>(results, avg_que, students, questions);
    row_sum<<<students / THREAD_COUNT, THREAD_COUNT>>>(results, avg_stud, questions);
  }

  // if input is students << questions, switch to rowsum without caching
}
