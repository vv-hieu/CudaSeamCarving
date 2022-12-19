#include "PnmUtils.h"

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

char* concatStr(const char* s1, const char* s2) {
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    if (result) {
        strcpy(result, s1);
        strcat(result, s2);
    }
    return result;
}

// Host

void seamCarving_host(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output, int outputWidth, int outputHeight) {
    
}

// Device

void seamCarving_device(const pixel_t* input, int inputWidth, int inputHeight, pixel_t* output, int outputWidth, int outputHeight) {

}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        printf("[ERROR] Invalid number of arguments\n");
        return 1; 
    }
    char* inputFile  = argv[1];
    char* outputFile = argv[3];
    int   newWidth   = atoi(argv[2]);
    int   blockSizeX = 64;
    int   blockSizeY = 64;

    for (char* c = outputFile; *c != '\0'; ++c) {
        if (*c == '.') {
            *c = '\0';
            break;
        }
    }

    if (argc == 6) {
        blockSizeX = atoi(argv[4]);
        blockSizeX = atoi(argv[5]);
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

    int outputWidth  = newWidth;
    int outputHeight = inputHeight;

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
