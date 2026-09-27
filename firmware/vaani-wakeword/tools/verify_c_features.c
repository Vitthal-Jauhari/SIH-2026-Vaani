#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mel_features.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_audio_16k_pcm16.raw> <output_features.bin>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        perror("Failed to open input PCM file");
        return 1;
    }

    int16_t audio[16000];
    size_t read_count = fread(audio, sizeof(int16_t), 16000, fin);
    fclose(fin);

    if (read_count < 16000) {
        fprintf(stderr, "Warning: Only read %zu / 16000 samples, zero padding remainder\n", read_count);
        for (size_t i = read_count; i < 16000; i++) audio[i] = 0;
    }

    mel_features_init();

    float c_features[TOTAL_FEATURES];
    mel_features_extract(audio, 16000, c_features);

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("Failed to open output features file");
        return 1;
    }
    fwrite(c_features, sizeof(float), TOTAL_FEATURES, fout);
    fclose(fout);

    printf("[SUCCESS] C feature extraction complete: %d floats written to %s\n", TOTAL_FEATURES, argv[2]);
    return 0;
}
