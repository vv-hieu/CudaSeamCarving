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

void computeEnergy_host(const pixel_t* input, int inputWidth, int inputHeight, float* output) {
    const float sobelXFilter[] = {
        -1.0f,  0.0f,  1.0f,
        -2.0f,  0.0f,  2.0f,
        -1.0f,  0.0f,  1.0f
    };
    const float sobelYFilter[] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };

    for (int x0 = 0; x0 < inputWidth; ++x0) {
        for (int y0 = 0; y0 < inputHeight; ++y0) {

            int i0 = x0 + y0 * inputWidth;

            float sobelX = 0.0f;
            float sobelY = 0.0f;

            for (int x1 = 0; x1 < 3; ++x1) {
                for (int y1 = 0; y1 < 3; ++y1) {
                    pixel_t pixel = input[clamp(x0 + x1 - 1, 0, inputWidth - 1) + clamp(y0 + y1 - 1, 0, inputHeight - 1) * inputWidth];
                    float val = 0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b;
                    int fIdx = x1 + y1 * 3;
                    sobelX += val * sobelXFilter[fIdx];
                    sobelY += val * sobelYFilter[fIdx];
                }
            }

            output[i0] = sqrt(sobelX * sobelX + sobelY * sobelY);
        }
    }
}

void findVerticalSeam_host(const float* energy, int inputWidth, int inputHeight, int* output) {
    float* energyToBottom = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path           = (int*)malloc(inputWidth * (inputHeight - 1) * sizeof(int));

    for (int col = 0; col < inputWidth; ++col) {
        int i = col + (inputHeight - 1) * inputWidth;
        energyToBottom[i] = energy[i];
    }

    float seamWeight = 0.0f;
    int   seamIdx    = -1;
    for (int row = inputHeight - 2; row >= 0; --row) {
        for (int col = 0; col < inputWidth; ++col) {
            int s = col - 1 < 0 ? 0 : col - 1;
            int t = col + 1 > inputWidth - 1 ? inputWidth - 1 : col + 1;
            
            int min = energyToBottom[s + (row + 1) * inputWidth];
            int idx = s;

            for (int i = s; i <= t; ++i) {
                int e = energyToBottom[i + (row + 1) * inputWidth];
                if (min > e) {
                    idx = i;
                    min = e;
                }
            }

            int i = col + row * inputWidth;
            energyToBottom[i] = energy[i] + min;
            path[i]           = idx;

            if (row == 0) {
                if (seamIdx < 0 || (seamIdx >= 0 && seamWeight > energyToBottom[i])) {
                    seamWeight = energyToBottom[i];
                    seamIdx    = col;
                }
            }
        }
    }

    output[0] = seamIdx;
    for (int row = 0; row < inputHeight - 1; ++row) {
        output[row + 1] = path[output[row] + row * inputWidth];
    }

    free(energyToBottom);
    free(path);
}

void removeVerticalSeam_host(const pixel_t* input, int inputWidth, int inputHeight, int* seam, pixel_t* output) {
    for (int row = 0; row < inputHeight; ++row) {
        for (int col = 0; col < inputWidth - 1; ++col) {
            output[col + row * (inputWidth - 1)] = input[col + (col >= seam[row]) + row * inputWidth];
        }
    }
}

void findHorizontalSeam_host(const float* energy, int inputWidth, int inputHeight, int* output) {
    float* energyToRight = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   path          = (int*)malloc((inputWidth - 1) * inputHeight * sizeof(int));

    for (int row = 0; row < inputHeight; ++row) {
        int i = inputWidth - 1 + row * inputWidth;
        energyToRight[i] = energy[i];
    }

    float seamWeight = 0.0f;
    int   seamIdx    = -1;
    for (int col = inputWidth - 2; col >= 0; --col) {
        for (int row = 0; row < inputHeight; ++row) {
            int s = row - 1 < 0 ? 0 : row - 1;
            int t = row + 1 > inputHeight - 1 ? inputHeight - 1 : row + 1;
            
            int min = energyToRight[col + 1 + s * inputWidth];
            int idx = s;

            for (int i = s; i <= t; ++i) {
                int e = energyToRight[col + 1 + i * inputWidth];
                if (min > e) {
                    idx = i;
                    min = e;
                }
            }

            int i = col + row * inputWidth;
            energyToRight[i] = energy[i] + min;
            path[col + row * (inputWidth - 1)] = idx;

            if (col == 0) {
                if (seamIdx < 0 || (seamIdx >= 0 && seamWeight > energyToRight[i])) {
                    seamWeight = energyToRight[i];
                    seamIdx    = row;
                }
            }
        }
    }

    output[0] = seamIdx;
    for (int col = 0; col < inputWidth - 1; ++col) {
        output[col + 1] = path[col + output[col] * (inputWidth - 1)];
    }

    free(energyToRight);
    free(path);
}

void removeHorizontalSeam_host(const pixel_t* input, int inputWidth, int inputHeight, int* seam, pixel_t* output) {
    for (int row = 0; row < inputHeight - 1; ++row) {
        for (int col = 0; col < inputWidth; ++col) {
            output[col + row * inputWidth] = input[col + (row + (row >= seam[col])) * inputWidth];
        }
    }
}

void seamCarving_host(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output, int outputWidth, int outputHeight, DebugInfo* debug) {
    bool outputDebug    = false;
    int  debugFileIndex = 0;
    char filename[1024];

    if (debug) {
        system("mkdir debug_info_host");
        outputDebug = true;
    }
    
    pixel_t* currentInput  = (pixel_t*)malloc(inputWidth * inputHeight * sizeof(pixel_t));
    pixel_t* currentOutput = (pixel_t*)malloc(inputWidth * inputHeight * sizeof(pixel_t));
    
    memcpy(currentOutput, input, inputWidth * inputHeight * sizeof(pixel_t));
    
    float* energy         = (float*)malloc(inputWidth * inputHeight * sizeof(float));
    int*   verticalSeam   = (int*)malloc(inputHeight * sizeof(int));
    int*   horizontalSeam = (int*)malloc(inputWidth * sizeof(int));

    // Remove vertical seams
    while (inputWidth > outputWidth) {

        pixel_t* temp = currentInput;
        currentInput  = currentOutput;
        currentOutput = temp;

        computeEnergy_host(currentInput, inputWidth, inputHeight, energy);

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

        findVerticalSeam_host(energy, inputWidth, inputHeight, verticalSeam);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int row = 0; row < inputHeight; ++row) {
                    fprintf(f, "%d %d\n", verticalSeam[row], row);
                }
                fclose(f);
            }
        }

        removeVerticalSeam_host(currentInput, inputWidth, inputHeight, verticalSeam, currentOutput);
        --inputWidth;
        ++debugFileIndex;
    }

    // Remove horizontal seams
    while (inputHeight > outputHeight) {
        pixel_t* temp = currentInput;
        currentInput  = currentOutput;
        currentOutput = temp;

        computeEnergy_host(currentInput, inputWidth, inputHeight, energy);

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

        findHorizontalSeam_host(energy, inputWidth, inputHeight, horizontalSeam);

        if (outputDebug) {
            sprintf(filename, "debug_info_host/%s_%d.txt", debug->outputSeamFile, debugFileIndex);
            FILE* f = fopen(filename, "w");
            if (f) {
                for (int col = 0; col < inputWidth; ++col) {
                    fprintf(f, "%d %d\n", col, horizontalSeam[col]);
                }
                fclose(f);
            }
        }

        removeHorizontalSeam_host(currentInput, inputWidth, inputHeight, horizontalSeam, currentOutput);
        --inputHeight;
        ++debugFileIndex;
    }

    memcpy(output, currentOutput, outputWidth * outputHeight * sizeof(pixel_t));

    free(currentInput);
    free(currentOutput);
    free(energy);
    free(verticalSeam);
    free(horizontalSeam);
}

// Device

void seamCarving_device(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output, int outputWidth, int outputHeight) {

}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7 && argc != 8 && argc != 10) {
        printf("[ERROR] Invalid number of arguments\n");
        return 1; 
    }
    char*      inputFile  = argv[1];
    char*      outputFile = argv[4];
    int        newWidth   = atoi(argv[2]);
    int        newHeight  = atoi(argv[3]);
    int        blockSizeX = 64;
    int        blockSizeY = 64;
    DebugInfo* debug      = nullptr;

    for (char* c = outputFile; *c != '\0'; ++c) {
        if (*c == '.') {
            *c = '\0';
            break;
        }
    }

    int d = 5;
    if (argc == 7 || argc == 10) {
        blockSizeX = atoi(argv[5]);
        blockSizeX = atoi(argv[6]);
        d = 7;
    }

    if (argc == 8 || argc == 10) {
        if (!strcmp(argv[d], "-d")) {
            debug = (DebugInfo*)malloc(sizeof(DebugInfo));
            debug->outputEnergyFile = argv[d + 1];
            debug->outputSeamFile   = argv[d + 2];
        }
        else {
            printf("[ERROR] Unknown argument %s\n", argv[d]);
            return 1;
        }
    }

    if (blockSizeX <= 0 || blockSizeY <= 0) {
        printf("[ERROR] Invalid arguments: Block size must be positive\n");
        return 1; 
    }

    int      inputWidth  = 0;
    int      inputHeight = 0;
    pixel_t* inputPixels = nullptr;
    if (readPnm(inputFile, inputPixels, inputWidth, inputHeight)) {
        return 1;
    }

    if (newWidth <= 0 || newWidth >= inputWidth) {
        printf("[ERROR] Invalid arguments: New width must be positive and smaller than input image width\n");
        return 1; 
    }

    if (newHeight <= 0 || newHeight >= inputHeight) {
        printf("[ERROR] Invalid arguments: New height must be positive and smaller than input image height\n");
        return 1; 
    }

    int outputWidth  = newWidth;
    int outputHeight = newHeight;

    // Info
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));

    printf("[INFO] GPU\n");
    printf("[INFO]     Name                         : %s\n"        , devProv.name);
    printf("[INFO]     Compute capability           : %d.%d\n"     , devProv.major, devProv.minor);
    printf("[INFO]     Number of SMs                : %d\n"        , devProv.multiProcessorCount);
    printf("[INFO]     Max number of threads per SM : %d\n"        , devProv.maxThreadsPerMultiProcessor); 
    printf("[INFO]     Max number of warps per SM   : %d\n"        , devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("[INFO]     GMEM                         : %zu byte\n"  , devProv.totalGlobalMem);
    printf("[INFO]     SMEM per SM                  : %zu byte\n"  , devProv.sharedMemPerMultiprocessor);
    printf("[INFO]     SMEM per block               : %zu byte\n\n", devProv.sharedMemPerBlock);
    printf("[INFO] Input image\n");
    printf("[INFO]     Width  : %d (pixels)\n"  , inputWidth);
    printf("[INFO]     Height : %d (pixels)\n\n", inputHeight);
    printf("[INFO] Output image\n");
    printf("[INFO]     Width  : %d (pixels)\n"  , outputWidth);
    printf("[INFO]     Height : %d (pixels)\n\n", outputHeight);

    if (debug) {
        printf("[INFO] Debug info\n");
        printf("[INFO]     Output energy file     : %s\n"  , debug->outputEnergyFile);
        printf("[INFO]     Output first seam file : %s\n\n", debug->outputSeamFile);
    }
    else {
        printf("[INFO] Debug info disabled\n\n");
    }

    char* hostOutputFile   = concatStr(outputFile, "_host.pnm");
    char* deviceOutputFile = concatStr(outputFile, "_device.pnm");

    pixel_t* hostOutputPixels   = (pixel_t*)malloc(outputWidth * outputHeight * sizeof(pixel_t));
    pixel_t* deviceOutputPixels = (pixel_t*)malloc(outputWidth * outputHeight * sizeof(pixel_t));
    
    if (!hostOutputPixels || !deviceOutputPixels) {
        printf("[ERROR] Unable to allocate memory for output\n");
        return 1;
    }

    GpuTimer timer;

    // Host
    seamCarving_host(inputPixels, inputWidth, inputHeight, hostOutputPixels, outputWidth, outputHeight, debug);
    writePnm(hostOutputFile, hostOutputPixels, outputWidth, outputHeight);

    // Device


    if (hostOutputPixels) {
        free(hostOutputPixels);
    }
    if (deviceOutputPixels) {
        free(deviceOutputPixels);
    }

    if (inputPixels) {
        free(inputPixels);
    }
    if (hostOutputFile) {
        free(hostOutputFile);
    }
    if (deviceOutputFile) {
        free(deviceOutputFile);
    }

    return 0;
}
