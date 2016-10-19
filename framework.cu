#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm> 
#include <cmath>

#include "kernel.cu"
#include "kernel_CPU.C"

#define STUDENTS  2048
#define QUESTIONS 1024
#define ITERS 1000

void generateRandomResults(int *results, int students, int questions) {
    int hardness[questions];
    for (int j = 0; j < questions; j++) {
        hardness[j] = rand() % 20;          // hard questions may decrease results by 20%
    }
    for (int i = 0; i < students; i++) {
        int iq = rand() % 20; // student's inteligence may add up to 20%
        for (int j = 0; j < questions; j++) {
            int r = std::max(rand() % 100 - hardness[j], 0); // 0-100 %
            results[i*questions + j] = std::min(r+iq, 100);  // 0-100 %
        }
    }
}

int main(int argc, char **argv){
    // Data for CPU computation
    int *results = NULL;    // students' results (input)
    float *avg_stud = NULL; // average per student (output)
    float *avg_que = NULL;  // average per question (output)
    // Data for GPU computation
    int *d_results = NULL;    // students' results in GPU memory (input)
    float *d_avg_stud = NULL; // average per student (output)
    float *d_avg_que = NULL;  // average per question (output)
    // CPU mirror of GPU results
    float *gpu_avg_stud = NULL; // average per student (output)
    float *gpu_avg_que = NULL;  // average per question (output)

    // parse command line
    int device = 0;
    if (argc == 2) 
        device = atoi(argv[1]);
    if (cudaSetDevice(device) != cudaSuccess){
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate and set host memory
    results = (int*)malloc(STUDENTS*QUESTIONS*sizeof(results[0]));
    avg_stud = (float*)malloc(STUDENTS*sizeof(avg_stud[0]));
    avg_que = (float*)malloc(QUESTIONS*sizeof(avg_stud[0]));
    generateRandomResults(results, STUDENTS, QUESTIONS);
    gpu_avg_stud = (float*)malloc(STUDENTS*sizeof(gpu_avg_stud[0]));
    gpu_avg_que = (float*)malloc(QUESTIONS*sizeof(gpu_avg_stud[0]));
 
    // allocate and set device memory
    if (cudaMalloc((void**)&d_results, STUDENTS*QUESTIONS*sizeof(d_results[0])) != cudaSuccess 
    || cudaMalloc((void**)&d_avg_stud, STUDENTS*sizeof(d_avg_stud[0])) != cudaSuccess
    || cudaMalloc((void**)&d_avg_que, QUESTIONS*sizeof(d_avg_que[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }

    // clean output arrays
    cudaMemset(d_avg_stud, 0, STUDENTS*sizeof(d_avg_stud[0]));
    cudaMemset(d_avg_que, 0, QUESTIONS*sizeof(d_avg_que[0]));


    // solve on CPU
    printf("Solving on CPU...\n");
    cudaEventRecord(start, 0);
    solveCPU(results, avg_stud, avg_que, STUDENTS, QUESTIONS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megaevals/s\n",
        float(STUDENTS*QUESTIONS)/time/1e3f);

    // copy data to GPU
    cudaMemcpy(d_results, results, STUDENTS*QUESTIONS*sizeof(d_results[0]), cudaMemcpyHostToDevice);

    // solve on GPU
    printf("Solving on GPU...\n");
    cudaEventRecord(start, 0);
    // iterate to improve measurement precision
    for (int i = 0; i < ITERS; i++)
        solveGPU(d_results, d_avg_stud, d_avg_que, STUDENTS, QUESTIONS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU performance: %f megaevals/s\n",
        float(STUDENTS*QUESTIONS)*float(ITERS)/time/1e3f);

    // check GPU results
    cudaMemcpy(gpu_avg_stud, d_avg_stud, STUDENTS*sizeof(d_avg_stud[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_avg_que, d_avg_que, QUESTIONS*sizeof(d_avg_que[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < STUDENTS; i++) {
        if (std::fabs(gpu_avg_stud[i] - avg_stud[i]) > 0.000001f) {
            printf("Error detected at index %i of avg_stud: %f should be %f.\n", i, gpu_avg_stud[i], avg_stud[i]);
            goto cleanup; // exit after first error
         }
    }
    for (int i = 0; i < QUESTIONS; i++) {
        if (std::fabs(gpu_avg_que[i] - avg_que[i]) > 0.000001f) {
            printf("Error detected at index %i of avg_que: %f should be %f.\n", i, gpu_avg_que[i], avg_que[i]);
            goto cleanup; // exit after first error
         }
    }

    printf("Test OK.\n");

cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (d_results) cudaFree(d_results);
    if (d_avg_stud) cudaFree(d_avg_stud);
    if (d_avg_que) cudaFree(d_avg_que);

    if (results) free(results);
    if (avg_stud) free(avg_stud);
    if (avg_que) free(avg_que);

    if (gpu_avg_stud) free(gpu_avg_stud);
    if (gpu_avg_que) free(gpu_avg_que);

    return 0;
}

