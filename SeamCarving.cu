#include "PnmUtils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(call) {\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        fprintf(stderr, "[CUDA ERROR] %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

struct DebugInfo {
    const char* outputEnergyFile;
    const char* outputSeamFile;
    const char* outputProfilerFile;
};

char* concatStr(const char* s1, const char* s2) {
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    if (result) {
        strcpy(result, s1);
        strcat(result, s2);
    }
    return result;
}

// Host

int clamp(int value, int min, int max) {
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

void grayscale_host(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output) {
    for (int x0 = 0; x0 < inputWidth; ++x0) {
        for (int y0 = 0; y0 < inputHeight; ++y0) {
            int idx = x0 + y0 * inputWidth;
            float grayscaleVal = 0.299f * input[idx].r + 0.587f * input[idx].g + 0.114f * input[idx].b;
            output[idx].r = grayscaleVal;
            output[idx].g = grayscaleVal;
            output[idx].b = grayscaleVal;
        }
    }
}

void computeEnergy_host(const pixel_t* input, int inputWidth, int inputHeight, float* output) {
    for (int x0 = 0; x0 < inputWidth; ++x0) {
        for (int y0 = 0; y0 < inputHeight; ++y0) {
            float sobelX = (
                -1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 - 1, 0, inputWidth - 1) + clamp(y0 - 1, 0, inputHeight - 1) * inputWidth]) +
                -2.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 - 1, 0, inputWidth - 1) + y0                                * inputWidth]) +
                -1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 - 1, 0, inputWidth - 1) + clamp(y0 + 1, 0, inputHeight - 1) * inputWidth]) +
                 1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 + 1, 0, inputWidth - 1) + clamp(y0 - 1, 0, inputHeight - 1) * inputWidth]) +
                 2.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 + 1, 0, inputWidth - 1) + y0                                * inputWidth]) +
                 1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 + 1, 0, inputWidth - 1) + clamp(y0 + 1, 0, inputHeight - 1) * inputWidth])
            );
            float sobelY = (
                -1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 - 1, 0, inputWidth - 1) + clamp(y0 - 1, 0, inputHeight - 1) * inputWidth]) +
                -2.0f * diff(input[x0 + y0 * inputWidth], input[x0                               + clamp(y0 - 1, 0, inputHeight - 1) * inputWidth]) +
                -1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 + 1, 0, inputWidth - 1) + clamp(y0 - 1, 0, inputHeight - 1) * inputWidth]) +
                 1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 - 1, 0, inputWidth - 1) + clamp(y0 + 1, 0, inputHeight - 1) * inputWidth]) +
                 2.0f * diff(input[x0 + y0 * inputWidth], input[x0                               + clamp(y0 + 1, 0, inputHeight - 1) * inputWidth]) +
                 1.0f * diff(input[x0 + y0 * inputWidth], input[clamp(x0 + 1, 0, inputWidth - 1) + clamp(y0 + 1, 0, inputHeight - 1) * inputWidth])
            );
            float val = sobelX * sobelX + sobelY * sobelY;
            output[x0 + y0 * inputWidth] = sqrtf(val < 0.0f ? 0.0f : val);
        }
    }
}

void findVerticalSeam_host(const float* energy, int inputWidth, int inputHeight, int* output, float* cumulativeEnergy, int* path) {
    for (int col = 0; col < inputWidth; ++col) {
        cumulativeEnergy[col] = energy[col];
    }

    for (int row = 1; row < inputHeight; ++row) {
        for (int col = 0; col < inputWidth; ++col) {
            int s = col - 1 < 0 ? 0 : col - 1;
            int t = col + 1 > inputWidth - 1 ? inputWidth - 1 : col + 1;
            
            int min = cumulativeEnergy[col + (row - 1) * inputWidth];
            int idx = col;

            for (int i = s; i <= t; ++i) {
                int e = cumulativeEnergy[i + (row - 1) * inputWidth];
                if (min > e) {
                    min = e;
                    idx = i;
                }
            }

            int i = col + row * inputWidth;
            cumulativeEnergy[i] = energy[i] + min;
            path[i]             = idx;
        }
    }

    float minSeamWeight = 0.0f;
    output[inputHeight - 1] = -1;

    int d = 32;
    int c = d * ((inputWidth - 1) / d + 1);
    for (int i = 0; i < c; ++i) {
        int idx = i / d + (i % d) * d;
        if (idx < inputWidth) {
            float e = cumulativeEnergy[idx + (inputHeight - 1) * inputWidth];
            if (output[inputHeight - 1] < 0 || (output[inputHeight - 1] >= 0 && minSeamWeight > e)) {
                minSeamWeight = e;
                output[inputHeight - 1] = idx;
            }
        }
    }

    for (int row = inputHeight - 2; row >= 0; --row) {
        output[row] = path[output[row + 1] + (row + 1) * inputWidth];
    }
}

void removeVerticalSeam_host(const pixel_t* input, int inputWidth, int inputHeight, const int* seam, pixel_t* output) {
    for (int row = 0; row < inputHeight; ++row) {
        for (int col = 0; col < inputWidth - 1; ++col) {
            output[col + row * (inputWidth - 1)] = input[col + (col >= seam[row]) + row * inputWidth];
        }
    }
}

void findHorizontalSeam_host(const float* energy, int inputWidth, int inputHeight, int* output, float* cumulativeEnergy, int* path) {
    for (int row = 0; row < inputHeight; ++row) {
        cumulativeEnergy[row] = energy[row];
    }

    for (int col = 1; col < inputWidth; ++col) {
        for (int row = 0; row < inputHeight; ++row) {
            int s = row - 1 < 0 ? 0 : row - 1;
            int t = row + 1 > inputHeight - 1 ? inputHeight - 1 : row + 1;
            
            int min = cumulativeEnergy[col - 1 + row * inputWidth];
            int idx = row;

            for (int i = s; i <= t; ++i) {
                int e = cumulativeEnergy[col - 1 + i * inputWidth];
                if (min > e) {
                    min = e;
                    idx = i;
                }
            }

            int i = col + row * inputWidth;
            cumulativeEnergy[i] = energy[i] + min;
            path[col + row * inputWidth] = idx;
        }
    }
    float minSeamWeight = 0.0f;
    output[inputHeight - 1] = -1;

    int d = 32;
    int c = d * ((inputHeight - 1) / d + 1);
    for (int i = 0; i < c; ++i) {
        int idx = i / d + (i % d) * d;
        if (idx < inputHeight) {
            float e = cumulativeEnergy[inputWidth - 1 + idx * inputWidth];
            if (output[inputWidth - 1] < 0 || (output[inputWidth - 1] >= 0 && minSeamWeight > e)) {
                minSeamWeight = e;
                output[inputWidth - 1] = idx;
            }
        }
    }

    for (int col = inputWidth - 2; col >= 0; --col) {
        output[col] = path[col + 1 + output[col + 1] * inputWidth];
    }
}

void removeHorizontalSeam_host(const pixel_t* input, int inputWidth, int inputHeight, const int* seam, pixel_t* output) {
    for (int row = 0; row < inputHeight - 1; ++row) {
        for (int col = 0; col < inputWidth; ++col) {
            output[col + row * inputWidth] = input[col + (row + (row >= seam[col])) * inputWidth];
        }
    }
}

void seamCarving_host(const image_t& input, image_t& output, DebugInfo* debug) {
    bool outputDebug    = false;
    int  debugFileIndex = 0;
    char filename[1024];

    if (debug) {
        system("rmdir -r debug_info_host");
        system("mkdir debug_info_host");
        outputDebug = true;
    }

    int inputWidth   = input.width;
    int inputHeight  = input.height;
    int outputWidth  = output.width;
    int outputHeight = output.height;
    
    pixel_t* currentInput  = (pixel_t*)malloc(inputWidth * inputHeight * sizeof(pixel_t));
    pixel_t* currentOutput = (pixel_t*)malloc(inputWidth * inputHeight * sizeof(pixel_t));
    
    memcpy(currentOutput, input.pixels, inputWidth * inputHeight * sizeof(pixel_t));
    
    pixel_t* grayscale      = (pixel_t*)malloc(inputWidth * inputHeight * sizeof(pixel_t));
    float* energy           = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    float* cumulativeEnergy = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path             = (int*)malloc(inputWidth * inputHeight * sizeof(int));
    int*   seam             = (int*)malloc((inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int));

    while (inputWidth > outputWidth) {

        pixel_t* temp = currentInput;
        currentInput  = currentOutput;
        currentOutput = temp;

        grayscale_host(currentInput, inputWidth, inputHeight, grayscale);

        computeEnergy_host(grayscale, inputWidth, inputHeight, energy);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        findVerticalSeam_host(energy, inputWidth, inputHeight, seam, cumulativeEnergy, path);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    fprintf(f, "%d %d\n", seam[row], row);
                }
                fclose(f);
            }
        }

        removeVerticalSeam_host(currentInput, inputWidth, inputHeight, seam, currentOutput);

        --inputWidth;
        ++debugFileIndex;
    }

    while (inputHeight > outputHeight) {
        pixel_t* temp = currentInput;
        currentInput  = currentOutput;
        currentOutput = temp;

        grayscale_host(currentInput, inputWidth, inputHeight, grayscale);

        computeEnergy_host(grayscale, inputWidth, inputHeight, energy);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        findHorizontalSeam_host(energy, inputWidth, inputHeight, seam, cumulativeEnergy, path);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int col = 0; col < inputWidth; ++col) {
                    fprintf(f, "%d %d\n", col, seam[col]);
                }
                fclose(f);
            }
        }

        removeHorizontalSeam_host(currentInput, inputWidth, inputHeight, seam, currentOutput);

        --inputHeight;
        ++debugFileIndex;
    }

    memcpy(output.pixels, currentOutput, outputWidth * outputHeight * sizeof(pixel_t));

    free(currentInput);
    free(currentOutput);
    free(grayscale);
    free(energy);
    free(cumulativeEnergy);
    free(path);
    free(seam);
}

// Device

__device__ int bCount0;
volatile __device__ int bCount1;

__device__ float pixelDiff_device(pixel_t p1, pixel_t p2) {
    float r = p2.r - p1.r;
    float g = p2.g - p1.g;
    float b = p2.b - p1.b;
    return (
        (r < 0.0f ? -r : r) +
        (g < 0.0f ? -g : g) +
        (b < 0.0f ? -b : b)
    );
}

__device__ int clamp_device(int value, int min, int max) {
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

__global__ void grayscale_device(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) {
        int idx = x + y * inputWidth;
        float grayscaleVal = 0.299f * input[idx].r + 0.587f * input[idx].g + 0.114f * input[idx].b;
        output[idx].r = grayscaleVal;
        output[idx].g = grayscaleVal;
        output[idx].b = grayscaleVal;
    }
}

__global__ void computeEnergy_device1(const pixel_t* input, int inputWidth, int inputHeight, float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) {
        float sobelX = (
            -1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x - 1, 0, inputWidth - 1) + clamp_device(y - 1, 0, inputHeight - 1) * inputWidth]) +
            -2.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x - 1, 0, inputWidth - 1) + y                                * inputWidth]) +
            -1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x - 1, 0, inputWidth - 1) + clamp_device(y + 1, 0, inputHeight - 1) * inputWidth]) +
             1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x + 1, 0, inputWidth - 1) + clamp_device(y - 1, 0, inputHeight - 1) * inputWidth]) +
             2.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x + 1, 0, inputWidth - 1) + y                                * inputWidth]) +
             1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x + 1, 0, inputWidth - 1) + clamp_device(y + 1, 0, inputHeight - 1) * inputWidth])
        );
        float sobelY = (
            -1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x - 1, 0, inputWidth - 1) + clamp_device(y - 1, 0, inputHeight - 1) * inputWidth]) +
            -2.0f * pixelDiff_device(input[x + y * inputWidth], input[x                                      + clamp_device(y - 1, 0, inputHeight - 1) * inputWidth]) +
            -1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x + 1, 0, inputWidth - 1) + clamp_device(y - 1, 0, inputHeight - 1) * inputWidth]) +
             1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x - 1, 0, inputWidth - 1) + clamp_device(y + 1, 0, inputHeight - 1) * inputWidth]) +
             2.0f * pixelDiff_device(input[x + y * inputWidth], input[x                                      + clamp_device(y + 1, 0, inputHeight - 1) * inputWidth]) +
             1.0f * pixelDiff_device(input[x + y * inputWidth], input[clamp_device(x + 1, 0, inputWidth - 1) + clamp_device(y + 1, 0, inputHeight - 1) * inputWidth])
        );
        float val = sobelX * sobelX + sobelY + sobelY;
        output[x + y * inputWidth] = sqrtf(val < 0.0f ? 0.0f : val);
    }
}

__global__ void computeEnergy_device2(const pixel_t* input, int inputWidth, int inputHeight, float* output) {
    extern __shared__ pixel_t sData[];
    int offset = 0;
    while (offset < (blockDim.x + 2) * (blockDim.y + 2)) {
        int dest = threadIdx.x + threadIdx.y * blockDim.x + offset;
        int destX = dest % (blockDim.x + 2);
        int destY = dest / (blockDim.x + 2);
        int srcX = blockIdx.x * blockDim.x + destX - 1;
        int srcY = blockIdx.y * blockDim.y + destY - 1;
        srcX = srcX < 0 ? 0 : srcX >= inputWidth  ? inputWidth  - 1 : srcX;
        srcY = srcY < 0 ? 0 : srcY >= inputHeight ? inputHeight - 1 : srcY;

        if (destY < blockDim.y + 2) {
            sData[dest] = input[srcX + srcY * inputWidth];
        }

        offset += blockDim.x * blockDim.y;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) {
        float sobelX = (
            -1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 0 + (threadIdx.y + 0) * (blockDim.x + 2)]) + 
            -2.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 0 + (threadIdx.y + 1) * (blockDim.x + 2)]) + 
            -1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 0 + (threadIdx.y + 2) * (blockDim.x + 2)]) + 
             1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 2 + (threadIdx.y + 0) * (blockDim.x + 2)]) + 
             2.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 2 + (threadIdx.y + 1) * (blockDim.x + 2)]) + 
             1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 2 + (threadIdx.y + 2) * (blockDim.x + 2)])
        );
        float sobelY = (
            -1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 0 + (threadIdx.y + 0) * (blockDim.x + 2)]) + 
            -2.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 1 + (threadIdx.y + 0) * (blockDim.x + 2)]) + 
            -1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 2 + (threadIdx.y + 0) * (blockDim.x + 2)]) + 
             1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 0 + (threadIdx.y + 2) * (blockDim.x + 2)]) + 
             2.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 1 + (threadIdx.y + 2) * (blockDim.x + 2)]) + 
             1.0f * pixelDiff_device(sData[threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2)], sData[threadIdx.x + 2 + (threadIdx.y + 2) * (blockDim.x + 2)])
        );
        float val = sobelX * sobelX + sobelY * sobelY;
        output[x + y * inputWidth] = sqrtf(val < 0.0f ? 0.0f : val);
    }
    __syncthreads();
}

__global__ void computeVerticalCumulativeEnergy_device(const float* energy, int inputWidth, int inputHeight, volatile float* cumulativeEnergy, int* path, int* pathIdx) {
    int k;
    
    extern __shared__ int sPathIdx[];

    __shared__ int bi;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        bi = atomicAdd(&bCount0, 1);
    }
    if (threadIdx.y == 0) {
        sPathIdx[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    k = 0;
    while (k < inputWidth) {
        int x = threadIdx.x + k;
        int y = threadIdx.y + bi * blockDim.y;

        if (x < inputWidth && y < inputHeight) {
            cumulativeEnergy[x + y * inputWidth] = energy[x + y * inputWidth];
        }

        k += blockDim.x;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        while (bCount1 < bi) {}
    }
    __syncthreads();

    for (int i = 0; i < blockDim.y; ++i) {
        if (i == threadIdx.y) {
            k = 0;
            while (k < inputWidth) {
                int x = threadIdx.x + k;
                int y = threadIdx.y + bi * blockDim.y;

                if (x < inputWidth && y < inputHeight && y > 0) {
                    int   sid = x - 1 < 0 ? 0 : x - 1;
                    int   tid = x + 1 > inputWidth - 1 ? inputWidth - 1 : x + 1;
                    float min = cumulativeEnergy[x + (y - 1) * inputWidth];
                    int   idx = x;

                    for (int j = sid; j <= tid; ++j) {
                        float e = cumulativeEnergy[j + (y - 1) * inputWidth];
                        if (min > e) {
                            min = e;
                            idx = j;
                        }
                    }

                    cumulativeEnergy[x + y * inputWidth] += min;
                    path[x + y * inputWidth] = idx;
                }

                k += blockDim.x;
            }
        }
        __syncthreads();
    }
    __threadfence();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ++bCount1;
    }
    __syncthreads();

    if (bi == gridDim.y - 1) {
        if (threadIdx.y == 0) {
            k = blockDim.x;
            while (k < inputWidth) {
                int x = threadIdx.x + k;
                if (x < inputWidth) {
                    if (cumulativeEnergy[sPathIdx[threadIdx.x] + (inputHeight - 1) * inputWidth] > cumulativeEnergy[x + (inputHeight - 1) * inputWidth]) {
                        sPathIdx[threadIdx.x] = x;
                    }
                }
                k += blockDim.x;
            }
        }
        __syncthreads();
        if (threadIdx.y == 0) {
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                if (threadIdx.x < inputWidth && threadIdx.x + stride < inputWidth) {
                    if (cumulativeEnergy[sPathIdx[threadIdx.x] + (inputHeight - 1) * inputWidth] > cumulativeEnergy[sPathIdx[threadIdx.x + stride] + (inputHeight - 1) * inputWidth]) {
                        sPathIdx[threadIdx.x] = sPathIdx[threadIdx.x + stride];
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            pathIdx[0] = sPathIdx[0];
        }
    }
}

__global__ void removeVerticalSeam_device(const pixel_t* input, int inputWidth, int inputHeight, const int* seam, pixel_t* output) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < inputWidth - 1 && y < inputHeight) {
        output[x + y * (inputWidth - 1)] = input[x + (x >= seam[y]) + y * inputWidth];
    }
}

__global__ void computeHorizontalCumulativeEnergy_device(const float* energy, int inputWidth, int inputHeight, volatile float* cumulativeEnergy, int* path, int* pathIdx) {
    int k;
    
    extern __shared__ int sPathIdx[];

    __shared__ int bi;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        bi = atomicAdd(&bCount0, 1);
    }
    if (threadIdx.x == 0) {
        sPathIdx[threadIdx.y] = threadIdx.y;
    }
    __syncthreads();

    k = 0;
    while (k < inputHeight) {
        int x = threadIdx.x + bi * blockDim.x;
        int y = threadIdx.y + k;

        if (x < inputWidth && y < inputHeight) {
            cumulativeEnergy[x + y * inputWidth] = energy[x + y * inputWidth];
        }

        k += blockDim.y;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        while (bCount1 < bi) {}
    }
    __syncthreads();

    for (int i = 0; i < blockDim.x; ++i) {
        if (i == threadIdx.x) {
            k = 0;
            while (k < inputHeight) {
                int x = threadIdx.x + bi * blockDim.x;
                int y = threadIdx.y + k;

                if (x < inputWidth && y < inputHeight && x > 0) {
                    int   sid = y - 1 < 0 ? 0 : y - 1;
                    int   tid = y + 1 > inputHeight - 1 ? inputHeight - 1 : y + 1;
                    float min = cumulativeEnergy[x - 1 + y * inputWidth];
                    int   idx = y;

                    for (int j = sid; j <= tid; ++j) {
                        float e = cumulativeEnergy[x - 1 + j * inputWidth];
                        if (min > e) {
                            min = e;
                            idx = j;
                        }
                    }

                    cumulativeEnergy[x + y * inputWidth] += min;
                    path[x + y * inputWidth] = idx;
                }

                k += blockDim.y;
            }
        }
        __syncthreads();
    }
    __threadfence();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ++bCount1;
    }
    __syncthreads();

    if (bi == gridDim.x - 1) {
        if (threadIdx.x == 0) {
            k = blockDim.y;
            while (k < inputHeight) {
                int y = threadIdx.y + k;
                if (y < inputHeight) {
                    if (cumulativeEnergy[inputWidth - 1 + sPathIdx[threadIdx.y] * inputWidth] > cumulativeEnergy[inputWidth - 1 + y * inputWidth]) {
                        sPathIdx[threadIdx.y] = y;
                    }
                }
                k += blockDim.y;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
                if (threadIdx.y < inputHeight && threadIdx.y + stride < inputHeight) {
                    if (cumulativeEnergy[inputWidth - 1 + sPathIdx[threadIdx.y] * inputWidth] > cumulativeEnergy[inputWidth - 1 + sPathIdx[threadIdx.y + stride] * inputWidth]) {
                        sPathIdx[threadIdx.y] = sPathIdx[threadIdx.y + stride];
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            pathIdx[0] = sPathIdx[0];
        }
    }
}

__global__ void removeHorizontalSeam_device(const pixel_t* input, int inputWidth, int inputHeight, const int* seam, pixel_t* output) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < inputWidth && y < inputHeight - 1) {
        output[x + y * inputWidth] = input[x + (y + (y >= seam[x])) * inputWidth];
    }
}

void seamCarving_device1(const image_t& input, image_t& output, DebugInfo* debug) {
    const int zero = 0;

    int inputWidth   = input.width;
    int inputHeight  = input.height;
    int outputWidth  = output.width;
    int outputHeight = output.height;

    dim3 blockSize;
    dim3 gridSize;

    bool outputDebug    = false;
    int  debugFileIndex = 0;
    char filename[1024];

    if (debug) {
        system("rmdir -r debug_info_device");
        system("mkdir debug_info_device");
        outputDebug = true;
    }

    float* energy           = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    float* cumulativeEnergy = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path             = (int*)malloc(inputWidth * inputHeight * sizeof(int));
    int*   seam             = (int*)malloc((inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int));

    pixel_t* d_currentInput;
    pixel_t* d_currentOutput;
    pixel_t* d_grayscale;
    float*   d_energy;
    float*   d_cumulativeEnergy;
    int*     d_path;
    int*     d_seam;
    int*     d_pathIdx;

    CHECK(cudaMalloc(&d_currentInput    , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_currentOutput   , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_grayscale       , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_energy          , inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_cumulativeEnergy, inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_path            , inputWidth * inputHeight * sizeof(int)));
    CHECK(cudaMalloc(&d_seam            , (inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int)));
    CHECK(cudaMalloc(&d_pathIdx         , sizeof(int)));

    CHECK(cudaMemcpy(d_currentOutput, input.pixels, inputWidth * inputHeight * sizeof(pixel_t), cudaMemcpyHostToDevice));

    while (inputWidth > outputWidth) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device1<<<
            gridSize, 
            blockSize
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3(1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeVerticalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.x * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputHeight - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputHeight - 2; i >= 0; --i) {
            seam[i] = path[seam[i + 1] + (i + 1) * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    fprintf(f, "%d %d\n", seam[row], row);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputHeight * sizeof(int), cudaMemcpyHostToDevice));
        removeVerticalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputWidth;
        ++debugFileIndex;
    }

    while (inputHeight > outputHeight) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device1<<<
            gridSize, 
            blockSize
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeHorizontalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.y * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputWidth - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputWidth - 2; i >= 0; --i) {
            seam[i] = path[i + 1 + seam[i + 1] * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int col = 0; col < inputWidth; ++col) {
                    fprintf(f, "%d %d\n", col, seam[col]);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputWidth * sizeof(int), cudaMemcpyHostToDevice));
        removeHorizontalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputHeight;
        ++debugFileIndex;
    }
    
    CHECK(cudaMemcpy(output.pixels, d_currentOutput, outputWidth * outputHeight * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_currentInput));
    CHECK(cudaFree(d_currentOutput));
    CHECK(cudaFree(d_grayscale));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_cumulativeEnergy));
    CHECK(cudaFree(d_path));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_pathIdx));

    free(energy);
    free(cumulativeEnergy);
    free(path);
    free(seam);
}

void seamCarving_device2(const image_t& input, image_t& output, DebugInfo* debug) {
    const int zero = 0;

    int inputWidth   = input.width;
    int inputHeight  = input.height;
    int outputWidth  = output.width;
    int outputHeight = output.height;

    dim3 blockSize;
    dim3 gridSize;

    bool outputDebug    = false;
    int  debugFileIndex = 0;
    char filename[1024];

    if (debug) {
        system("rmdir -r debug_info_device");
        system("mkdir debug_info_device");
        outputDebug = true;
    }

    float* energy           = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    float* cumulativeEnergy = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path             = (int*)malloc(inputWidth * inputHeight * sizeof(int));
    int*   seam             = (int*)malloc((inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int));

    pixel_t* d_currentInput;
    pixel_t* d_currentOutput;
    pixel_t* d_grayscale;
    float*   d_energy;
    float*   d_cumulativeEnergy;
    int*     d_path;
    int*     d_seam;
    int*     d_pathIdx;

    CHECK(cudaMalloc(&d_currentInput    , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_currentOutput   , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_grayscale       , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_energy          , inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_cumulativeEnergy, inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_path            , inputWidth * inputHeight * sizeof(int)));
    CHECK(cudaMalloc(&d_seam            , (inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int)));
    CHECK(cudaMalloc(&d_pathIdx         , sizeof(int)));

    CHECK(cudaMemcpy(d_currentOutput, input.pixels, inputWidth * inputHeight * sizeof(pixel_t), cudaMemcpyHostToDevice));

    while (inputWidth > outputWidth) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device1<<<
            gridSize, 
            blockSize
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(1024, 1);
        gridSize = dim3(1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeVerticalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.x * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputHeight - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputHeight - 2; i >= 0; --i) {
            seam[i] = path[seam[i + 1] + (i + 1) * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    fprintf(f, "%d %d\n", seam[row], row);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputHeight * sizeof(int), cudaMemcpyHostToDevice));
        removeVerticalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputWidth;
        ++debugFileIndex;
    }

    while (inputHeight > outputHeight) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device1<<<
            gridSize, 
            blockSize
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(1, 1024);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeHorizontalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.y * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputWidth - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputWidth - 2; i >= 0; --i) {
            seam[i] = path[i + 1 + seam[i + 1] * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int col = 0; col < inputWidth; ++col) {
                    fprintf(f, "%d %d\n", col, seam[col]);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputWidth * sizeof(int), cudaMemcpyHostToDevice));
        removeHorizontalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputHeight;
        ++debugFileIndex;
    }
    
    CHECK(cudaMemcpy(output.pixels, d_currentOutput, outputWidth * outputHeight * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_currentInput));
    CHECK(cudaFree(d_currentOutput));
    CHECK(cudaFree(d_grayscale));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_cumulativeEnergy));
    CHECK(cudaFree(d_path));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_pathIdx));

    free(energy);
    free(cumulativeEnergy);
    free(path);
    free(seam);
}

void seamCarving_device3(const image_t& input, image_t& output, DebugInfo* debug) {
    const int zero = 0;

    int inputWidth   = input.width;
    int inputHeight  = input.height;
    int outputWidth  = output.width;
    int outputHeight = output.height;

    dim3 blockSize;
    dim3 gridSize;

    bool outputDebug    = false;
    int  debugFileIndex = 0;
    char filename[1024];

    if (debug) {
        system("rmdir -r debug_info_device");
        system("mkdir debug_info_device");
        outputDebug = true;
    }

    float* energy           = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    float* cumulativeEnergy = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path             = (int*)malloc(inputWidth * inputHeight * sizeof(int));
    int*   seam             = (int*)malloc((inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int));

    pixel_t* d_currentInput;
    pixel_t* d_currentOutput;
    pixel_t* d_grayscale;
    float*   d_energy;
    float*   d_cumulativeEnergy;
    int*     d_path;
    int*     d_seam;
    int*     d_pathIdx;

    CHECK(cudaMalloc(&d_currentInput    , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_currentOutput   , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_grayscale       , inputWidth * inputHeight * sizeof(pixel_t)));
    CHECK(cudaMalloc(&d_energy          , inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_cumulativeEnergy, inputWidth * inputHeight * sizeof(float)));
    CHECK(cudaMalloc(&d_path            , inputWidth * inputHeight * sizeof(int)));
    CHECK(cudaMalloc(&d_seam            , (inputWidth < inputHeight ? inputHeight : inputWidth) * sizeof(int)));
    CHECK(cudaMalloc(&d_pathIdx         , sizeof(int)));

    CHECK(cudaMemcpy(d_currentOutput, input.pixels, inputWidth * inputHeight * sizeof(pixel_t), cudaMemcpyHostToDevice));

    while (inputWidth > outputWidth) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device2<<<
            gridSize, 
            blockSize,
            (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel_t)
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(1024, 1);
        gridSize = dim3(1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeVerticalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.x * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputHeight - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputHeight - 2; i >= 0; --i) {
            seam[i] = path[seam[i + 1] + (i + 1) * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    fprintf(f, "%d %d\n", seam[row], row);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputHeight * sizeof(int), cudaMemcpyHostToDevice));
        removeVerticalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputWidth;
        ++debugFileIndex;
    }

    while (inputHeight > outputHeight) {
        pixel_t* temp   = d_currentInput;
        d_currentInput  = d_currentOutput;
        d_currentOutput = temp;

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        grayscale_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_grayscale);
        CHECK(cudaPeekAtLastError());

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        computeEnergy_device2<<<
            gridSize, 
            blockSize,
            (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel_t)
        >>>(d_grayscale, inputWidth, inputHeight, d_energy);
        CHECK(cudaPeekAtLastError());

        if (outputDebug) {
            CHECK(cudaMemcpy(energy, d_energy, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));

            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputEnergyFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    for (int col = 0; col < inputWidth; ++col) {
                        int i = row * inputWidth + col;
                        if (col != inputWidth - 1) {
                            fprintf(f, "%.3f ", energy[i]);
                        }
                        else {
                            fprintf(f, "%.3f\n", energy[i]);
                        }
                    }
                }
                fclose(f);
            }
        }

        blockSize = dim3(1, 1024);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, 1);
        CHECK(cudaMemcpyToSymbol(bCount0, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));
        computeHorizontalCumulativeEnergy_device<<<
            gridSize,
            blockSize,
            blockSize.y * sizeof(int)
        >>>(d_energy, inputWidth, inputHeight, d_cumulativeEnergy, d_path, d_pathIdx);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaMemcpy(path, d_path, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(seam + inputWidth - 1, d_pathIdx, sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = inputWidth - 2; i >= 0; --i) {
            seam[i] = path[i + 1 + seam[i + 1] * inputWidth];
        }

        if (outputDebug) {
            sprintf(filename, "debug_info_device/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int col = 0; col < inputWidth; ++col) {
                    fprintf(f, "%d %d\n", col, seam[col]);
                }
                fclose(f);
            }
        }

        blockSize = dim3(32, 32);
        gridSize = dim3((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
        CHECK(cudaMemcpy(d_seam, seam, inputWidth * sizeof(int), cudaMemcpyHostToDevice));
        removeHorizontalSeam_device<<<
            gridSize,
            blockSize
        >>>(d_currentInput, inputWidth, inputHeight, d_seam, d_currentOutput);
        CHECK(cudaPeekAtLastError());

        --inputHeight;
        ++debugFileIndex;
    }
    
    CHECK(cudaMemcpy(output.pixels, d_currentOutput, outputWidth * outputHeight * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_currentInput));
    CHECK(cudaFree(d_currentOutput));
    CHECK(cudaFree(d_grayscale));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_cumulativeEnergy));
    CHECK(cudaFree(d_path));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_pathIdx));

    free(energy);
    free(cumulativeEnergy);
    free(path);
    free(seam);
}

int main(int argc, char** argv) {

    if (argc != 6 && argc != 10) {
        printf("[ERROR] Invalid number of arguments\n");
        return 1; 
    }

    int        version    = atoi(argv[1]);
    char*      inputFile  = argv[2];
    char*      outputFile = argv[5];
    int        newWidth   = atoi(argv[3]);
    int        newHeight  = atoi(argv[4]);
    DebugInfo* debug      = nullptr;

    for (char* c = outputFile; *c != '\0'; ++c) {
        if (*c == '.') {
            *c = '\0';
            break;
        }
    }

    if (argc == 10) {
        if (!strcmp(argv[6], "-d")) {
            debug = (DebugInfo*)malloc(sizeof(DebugInfo));
            debug->outputEnergyFile   = argv[7];
            debug->outputSeamFile     = argv[8];
            debug->outputProfilerFile = argv[9];
        }
        else {
            printf("[ERROR] Unknown argument %s\n", argv[6]);
            return 1;
        }
    }

    image_t inputImage;
    if (readImage(inputFile, inputImage)) {
        return 1;
    }

    if (newWidth <= 0 || newWidth > inputImage.width) {
        printf("[ERROR] Invalid arguments: New width must be positive and smaller than input image width\n");
        return 1; 
    }

    if (newHeight <= 0 || newHeight > inputImage.height) {
        printf("[ERROR] Invalid arguments: New height must be positive and smaller than input image height\n");
        return 1; 
    }

    GpuTimer timer;

    // Info
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));

    printf("[INFO] ====================== GPU info ======================\n");
    printf("[INFO]     Name                         : %s\n"      , devProv.name);
    printf("[INFO]     Compute capability           : %d.%d\n"   , devProv.major, devProv.minor);
    printf("[INFO]     Number of SMs                : %d\n"      , devProv.multiProcessorCount);
    printf("[INFO]     Max number of threads per SM : %d\n"      , devProv.maxThreadsPerMultiProcessor); 
    printf("[INFO]     Max number of warps per SM   : %d\n"      , devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("[INFO]     GMEM                         : %zu byte\n", devProv.totalGlobalMem);
    printf("[INFO]     SMEM per SM                  : %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("[INFO]     SMEM per block               : %zu byte\n", devProv.sharedMemPerBlock);

    printf("[INFO] ==================== Input image =====================\n");
    printf("[INFO]     Width  : %d (pixels)\n", inputImage.width);
    printf("[INFO]     Height : %d (pixels)\n", inputImage.height);

    printf("[INFO] ==================== Output image ==================== \n");
    printf("[INFO]     Width  : %d (pixels)\n", newWidth);
    printf("[INFO]     Height : %d (pixels)\n", newHeight);

    printf("[INFO] ===================== Debug info ===================== \n");
    if (debug) {
        printf("[INFO]     Output energy file   : %s\n", debug->outputEnergyFile);
        printf("[INFO]     Output seam file     : %s\n", debug->outputSeamFile);
        printf("[INFO]     Output profiler file : %s\n", debug->outputProfilerFile);
    }
    else {
        printf("[INFO]     Disabled\n");
    }

    printf("[INFO] ================== Execution info ==================== \n");

    // Host
    char* hostOutputFile = concatStr(outputFile, "_host.pnm");
    image_t hostOutputImage;
    if (allocateImage(hostOutputImage, newWidth, newHeight)) {
        printf("[ERROR] Unable to allocate space for output\n");
        return 1;
    }
    timer.Start();
    seamCarving_host(inputImage, hostOutputImage, debug);
    timer.Stop();
    printf("[INFO]     Host implementation\n");
    printf("[INFO]         Execution time : %.3f ms\n", timer.Elapsed());
    writeImage(hostOutputFile, hostOutputImage);
    
    if (version == 1) {
        // Device 1
        char* device1OutputFile = concatStr(outputFile, "_device1.pnm");
        image_t device1OutputImage;
        if (allocateImage(device1OutputImage, newWidth, newHeight)) {
            printf("[ERROR] Unable to allocate space for output\n");
            return 1;
        }
        timer.Start();
        seamCarving_device1(inputImage, device1OutputImage, debug);
        timer.Stop();
        printf("[INFO]     Device 1 implementation\n");
        printf("[INFO]         Execution time : %.3f ms\n", timer.Elapsed());
        printf("[INFO]         Error          : %.3f\n", computeError(hostOutputImage.pixels, device1OutputImage.pixels, newWidth * newHeight));
        writeImage(device1OutputFile, device1OutputImage);
        freeImage(device1OutputImage);
        if (device1OutputFile) {
            free(device1OutputFile);
        }
    }
    else if (version == 2) {        
        // Device 2
        char* device2OutputFile = concatStr(outputFile, "_device2.pnm");
        image_t device2OutputImage;
        if (allocateImage(device2OutputImage, newWidth, newHeight)) {
            printf("[ERROR] Unable to allocate space for output\n");
            return 1;
        }
        timer.Start();
        seamCarving_device2(inputImage, device2OutputImage, debug);
        timer.Stop();
        printf("[INFO]     Device 2 implementation\n");
        printf("[INFO]         Execution time : %.3f ms\n", timer.Elapsed());
        printf("[INFO]         Error          : %.3f\n", computeError(hostOutputImage.pixels, device2OutputImage.pixels, newWidth * newHeight));
        writeImage(device2OutputFile, device2OutputImage);
        freeImage(device2OutputImage);
        if (device2OutputFile) {
            free(device2OutputFile);
        }
    }
    else if (version == 3) {
        // Device 3
        char* device3OutputFile = concatStr(outputFile, "_device3.pnm");
        image_t device3OutputImage;
        if (allocateImage(device3OutputImage, newWidth, newHeight)) {
            printf("[ERROR] Unable to allocate space for output\n");
            return 1;
        }
        timer.Start();
        seamCarving_device3(inputImage, device3OutputImage, debug);
        timer.Stop();
        printf("[INFO]     Device 3 implementation\n");
        printf("[INFO]         Execution time : %.3f ms\n", timer.Elapsed());
        printf("[INFO]         Error          : %.3f\n", computeError(hostOutputImage.pixels, device3OutputImage.pixels, newWidth * newHeight));
        writeImage(device3OutputFile, device3OutputImage);
        freeImage(device3OutputImage);
        if (device3OutputFile) {
            free(device3OutputFile);
        }
    }
    
    if (debug) {
        free(debug);
    }

    freeImage(hostOutputImage);
    if (hostOutputFile) {
        free(hostOutputFile);
    }

    return 0;
}
