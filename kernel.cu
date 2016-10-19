// CUDA-C includes
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

const int BLOCK = 32;

__global__ void col_sum(const int *in, float *avg_que, const size_t students, const size_t questions) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_que = 0;

    for (size_t j = 0; j < students; ++j) {
        tmp_que += in[questions*j + i];
    }

    avg_que[i] = float(tmp_que) / float(students);
}

__global__ void row_sum(const int *in, float *avg_stud, const size_t questions) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_stud = 0;

    for (size_t j = 0; j < questions; ++j) {
        tmp_stud += in[questions*i + j];
    }

    avg_stud[i] = float(tmp_stud) / float(questions);
}

void solveGPU(const int *results, float *avg_stud, float *avg_que, const size_t students, const size_t questions){
    col_sum<<<questions/BLOCK, BLOCK>>>(results, avg_que, students, questions);
    row_sum<<<students/BLOCK, BLOCK>>>(results, avg_stud, questions);
}

