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

struct image_t {
    pixel_t* pixels;
    int width;
    int height;
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
	err /= n * 3.0f;
	return err;
}

float diff(pixel_t p1, pixel_t p2) {
    float r = p2.r - p1.r;
    float g = p2.g - p1.g;
    float b = p2.b - p1.b;
    return 0.3f * r + 0.59f * g + 0.11f * b;
}

int fpeek(FILE *stream)
{
    int c;
    c = fgetc(stream);
    ungetc(c, stream);
    return c;
}

int readImage(const char* fileName, image_t& image) {
	FILE * f = fopen(fileName, "rb");
	if (!f) {
		printf("[ERROR] Unable to read %s\n", fileName);
		return 1;
	}

	char type[3];
	fscanf(f, "%s", type);

    if (!strcmp(type, "P1")) {
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
	    for (int i = 0; i < image.width * image.height; i++) {
            int vi;
		    fscanf(f, "%i", &vi);
            image.pixels[i] = makePixel(vi, vi, vi);
        }
    }
    else if (!strcmp(type, "P4")) {
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);
        
        bool cleaned = false;
        while (!cleaned) {
            cleaned = true;
            while (fpeek(f) == ' ') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\n') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\r') {
                fgetc(f);
                cleaned = false;
            }
        }

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
	    unsigned char* buffer = (unsigned char*)malloc(image.width * image.height);
        int bit = 7;
        int idx = 0;
        fread(buffer, image.width * image.height, 1, f);
        for (int i = 0; i < image.width * image.height; i++) {
            int vi = (buffer[idx] >> bit) & 1;
            image.pixels[i] = makePixel(vi, vi, vi);
            --bit;
            if (bit < 0) {
                bit = 7;
                ++idx;
            }
        }
        free(buffer);
    }
	else if (!strcmp(type, "P2")) {
        int maxVal;
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
	    for (int i = 0; i < image.width * image.height; i++) {
            int vi;
		    fscanf(f, "%i", &vi);
            float vi2 = vi * invMaxVal;
            image.pixels[i] = makePixel(vi2, vi2, vi2);
        }
    }
    else if (!strcmp(type, "P5")) {
        int maxVal;
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;
        
        bool cleaned = false;
        while (!cleaned) {
            cleaned = true;
            while (fpeek(f) == ' ') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\n') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\r') {
                fgetc(f);
                cleaned = false;
            }
        }

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
        unsigned char* buffer = (unsigned char*)malloc(image.width * image.height);
        fread(buffer, image.width * image.height, 1, f);
        for (int i = 0; i < image.width * image.height; i++) {
            float vi = buffer[i] * invMaxVal;
            image.pixels[i] = makePixel(vi, vi, vi);
        }
        free(buffer);
    }
	else if (!strcmp(type, "P3")) {
        int maxVal;
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
	    for (int i = 0; i < image.width * image.height; i++) {
            int ri, gi, bi;
		    fscanf(f, "%i%i%i", &ri, &gi, &bi);
            image.pixels[i] = makePixel(
                ri * invMaxVal,
                gi * invMaxVal,
                bi * invMaxVal
            );
        }
	}
    else if (!strcmp(type, "P6")){
        int maxVal;
        fscanf(f, "%i", &image.width);
	    fscanf(f, "%i", &image.height);
	    fscanf(f, "%i", &maxVal);
        float invMaxVal = 1.0f / maxVal;
        
        bool cleaned = false;
        while (!cleaned) {
            cleaned = true;
            while (fpeek(f) == ' ') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\n') {
                fgetc(f);
                cleaned = false;
            }
            while (fpeek(f) == '\r') {
                fgetc(f);
                cleaned = false;
            }
        }

        image.pixels = (pixel_t*)malloc(image.width * image.height * sizeof(pixel_t));
        unsigned char* buffer = (unsigned char*)malloc(image.width * image.height * 3);
        fread(buffer, image.width * image.height * 3, 1, f);
	    for (int i = 0; i < image.width * image.height; i++) {
            image.pixels[i] = makePixel(
                buffer[3 * i + 0] * invMaxVal,
                buffer[3 * i + 1] * invMaxVal,
                buffer[3 * i + 2] * invMaxVal
            );
        }
        free(buffer);
    }
    else {
        printf("[ERROR] Unsupported PNM type %s\n", type);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

int writeImage(const char* fileName, const image_t& image) {
	FILE * f = fopen(fileName, "w");
	if (!f) {
		printf("[ERROR] Unable to write %s\n", fileName);
        return 1;
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", image.width, image.height); 

	for (int i = 0; i < image.width * image.height; i++) {
        int ri = 255 * image.pixels[i].r;
        int gi = 255 * image.pixels[i].g;
        int bi = 255 * image.pixels[i].b;
        fprintf(f, "%i %i %i\n", ri, gi, bi);
    }
		
	fclose(f);
    return 0;
}

int allocateImage(image_t& image, int width, int height) {
    pixel_t* pixels = (pixel_t*)malloc(width * height * sizeof(pixel_t));
    if (!pixels) {
        image.pixels = nullptr;
        image.width  = 0;
        image.height = 0;
        return 1;
    }

    image.pixels = pixels;
    image.width  = width;
    image.height = height;
    return 0;
}

int freeImage(image_t& image) {
    if (image.pixels) {
        free(image.pixels);
        image.pixels = nullptr;
    }
    image.width  = 0;
    image.height = 0;
    return 0;
}
