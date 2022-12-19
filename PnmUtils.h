#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct pixel_t {
    float r; // Value range [0.0, 1.0]
    float g; // Value range [0.0, 1.0]
    float b; // Value range [0.0, 1.0]
};

pixel_t makePixel(float r, float g, float b) {
    pixel_t res;
    res.r = r;
    res.g = g;
    res.b = b;
    return res;
}

float computeError(const pixel_t* p1, const pixel_t* p2, int n) {
    float err = 0.0f;
	for (int i = 0; i < n; i++)
	{
		err += p1[i].r - p2[i].r > 0.0f ? p1[i].r - p2[i].r : p2[i].r - p1[i].r;
		err += p1[i].g - p2[i].g > 0.0f ? p1[i].g - p2[i].g : p2[i].g - p1[i].g;
		err += p1[i].b - p2[i].b > 0.0f ? p1[i].b - p2[i].b : p2[i].b - p1[i].b;
	}
	err *= 255.0f / (n * 3.0f);
	return err;
}

int readPnm(const char* fileName, pixel_t*& pixels, int& width, int& height) {
	FILE * f = fopen(fileName, "r");
	if (!f) {
		printf("[ERROR] Unable to read %s\n", fileName);
		return 1;
	}

	char type[3];
	fscanf(f, "%s", type);

	if (!strcmp(type, "P2")) {
        int maxVal;
        fscanf(f, "%i", &width);
	    fscanf(f, "%i", &height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;

        pixels = (pixel_t*)malloc(width * height * sizeof(pixel_t));
	    for (int i = 0; i < width * height; i++) {
            int vi;
		    fscanf(f, "%i", &vi);
            pixels[i] = makePixel(
                vi * invMaxVal,
                vi * invMaxVal,
                vi * invMaxVal
            );
        }
    }
	else if (!strcmp(type, "P3")) {
        int maxVal;
        fscanf(f, "%i", &width);
	    fscanf(f, "%i", &height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;

        pixels = (pixel_t*)malloc(width * height * sizeof(pixel_t));
	    for (int i = 0; i < width * height; i++) {
            int ri, gi, bi;
		    fscanf(f, "%i%i%i", &ri, &gi, &bi);
            pixels[i] = makePixel(
                ri * invMaxVal,
                gi * invMaxVal,
                bi * invMaxVal
            );
        }
	}
    else {
        printf("[ERROR] Unsupported PNM type %s\n", type);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

int writePnm(const char* fileName, pixel_t* pixels, int width, int height) {
	FILE * f = fopen(fileName, "w");
	if (!f) {
		printf("[ERROR] Unable to write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++) {
        int ri = 255 * pixels[i].r;
        int gi = 255 * pixels[i].g;
        int bi = 255 * pixels[i].b;
        fprintf(f, "%i\n%i\n%i\n", ri, gi, bi);
    }
		
	fclose(f);
    return 0;
}