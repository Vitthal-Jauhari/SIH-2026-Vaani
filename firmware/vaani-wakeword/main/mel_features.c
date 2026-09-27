/**
 * @file mel_features.c
 * @brief On-device MFCC feature extraction for V2 DS-CNN Wakeword Model
 *
 * Computes 63 frames × 13 MFCC coefficients matching Librosa:
 *   librosa.feature.mfcc(sr=16000, n_mfcc=13, n_mels=40, n_fft=512, hop_length=256, center=True)
 */

#include "mel_features.h"
#include "mel_tables.h"

#include <math.h>
#include <string.h>
#include "esp_log.h"

static const char *TAG = "MFCC_FEAT";

#define SAMPLE_RATE     16000
#define FFT_SIZE        512
#define HOP_LEN         256
#define N_MELS          40
#define NUM_FFT_BINS    (FFT_SIZE / 2 + 1)  /* 257 */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* FFT working buffer: interleaved [re, im, re, im, …] */
static float s_fft_buf[FFT_SIZE * 2];

/* Power spectrum for one frame */
static float s_power[NUM_FFT_BINS];

/* Full log-mel spectrogram [mel_bin * NUM_FRAMES + frame] */
static float s_log_mel[N_MELS * MEL_NUM_FRAMES];


/* ================================================================
 * Radix-2 Cooley–Tukey FFT (in-place, complex interleaved)
 * ================================================================ */

static void fft_bit_reverse(float *data, int n)
{
    int j = 0;

    for (int i = 0; i < n; i++) {
        if (j > i) {
            float tmp;
            tmp = data[2 * i];
            data[2 * i] = data[2 * j];
            data[2 * j] = tmp;

            tmp = data[2 * i + 1];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j + 1] = tmp;
        }

        int m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

static void fft_compute(float *data, int n)
{
    fft_bit_reverse(data, n);

    for (int step = 1; step < n; step <<= 1) {

        float theta = -M_PI / (float)step;
        float w_re  = cosf(theta);
        float w_im  = sinf(theta);

        for (int group = 0; group < n; group += (step << 1)) {

            float tw_re = 1.0f;
            float tw_im = 0.0f;

            for (int pair = 0; pair < step; pair++) {

                int i = group + pair;
                int j = i + step;

                float t_re = data[2 * j]     * tw_re
                           - data[2 * j + 1] * tw_im;

                float t_im = data[2 * j]     * tw_im
                           + data[2 * j + 1] * tw_re;

                data[2 * j]     = data[2 * i]     - t_re;
                data[2 * j + 1] = data[2 * i + 1] - t_im;
                data[2 * i]     += t_re;
                data[2 * i + 1] += t_im;

                float new_tw_re = tw_re * w_re - tw_im * w_im;
                tw_im           = tw_re * w_im + tw_im * w_re;
                tw_re           = new_tw_re;
            }
        }
    }
}


/* ================================================================
 * Initialization
 * ================================================================ */

void mel_features_init(void)
{
    ESP_LOGI(TAG, "MFCC feature extractor initialized.");
    ESP_LOGI(TAG, "  FFT=%d, Hop=%d, Mels=%d, Frames=%d, MFCCs=%d (total=%d)",
             FFT_SIZE, HOP_LEN, N_MELS, MEL_NUM_FRAMES, MEL_NUM_MFCC, TOTAL_FEATURES);
}


/* ================================================================
 * Feature Extraction
 * ================================================================ */

void mel_features_extract(const int16_t *audio, size_t num_samples,
                          float *output)
{
    /* ----------------------------------------------------------
     * 1. Compute 63 STFT frames with center=True (pad = 256)
     * ---------------------------------------------------------- */

    for (int frame = 0; frame < MEL_NUM_FRAMES; frame++) {

        /* Frame center is at (frame * HOP_LEN).
         * With pad = FFT_SIZE / 2 = 256, window start in original audio is: */
        int start_idx = (frame * HOP_LEN) - (FFT_SIZE / 2);

        memset(s_fft_buf, 0, sizeof(s_fft_buf));

        for (int i = 0; i < FFT_SIZE; i++) {
            int audio_idx = start_idx + i;
            float sample = 0.0f;

            if (audio_idx >= 0 && audio_idx < (int)num_samples) {
                sample = (float)audio[audio_idx] / 32768.0f;
            }

            s_fft_buf[2 * i] = sample * s_hann_window[i];
            /* Imaginary part stays 0 */
        }

        /* Compute FFT */
        fft_compute(s_fft_buf, FFT_SIZE);

        /* Compute Power Spectrum |X[k]|^2 */
        for (int k = 0; k < NUM_FFT_BINS; k++) {
            float re = s_fft_buf[2 * k];
            float im = s_fft_buf[2 * k + 1];
            s_power[k] = re * re + im * im;
        }

        /* Filter into 40 Mel bands and take log-dB */
        for (int m = 0; m < N_MELS; m++) {
            float energy = 0.0f;
            int start  = s_filters[m].start_bin;
            int count  = s_filters[m].num_weights;
            int offset = s_filters[m].weight_offset;

            for (int j = 0; j < count; j++) {
                int bin = start + j;
                if (bin < NUM_FFT_BINS) {
                    energy += s_power[bin] * s_mel_weights[offset + j];
                }
            }

            /* 10 * log10(max(1e-10, energy)) */
            float db = 10.0f * log10f(fmaxf(energy, 1e-10f));
            s_log_mel[m * MEL_NUM_FRAMES + frame] = db;
        }
    }

    /* ----------------------------------------------------------
     * 2. Apply top_db = 80.0 dynamic range clamp (Librosa power_to_db)
     * ---------------------------------------------------------- */

    float max_db = -1000.0f;
    for (int i = 0; i < N_MELS * MEL_NUM_FRAMES; i++) {
        if (s_log_mel[i] > max_db) {
            max_db = s_log_mel[i];
        }
    }

    float min_db = max_db - 80.0f;
    for (int i = 0; i < N_MELS * MEL_NUM_FRAMES; i++) {
        if (s_log_mel[i] < min_db) {
            s_log_mel[i] = min_db;
        }
    }

    /* ----------------------------------------------------------
     * 3. Apply DCT-II (ortho) -> 13 MFCCs per frame
     *
     * Output shape: [frame][mfcc] -> 63 x 13
     * ---------------------------------------------------------- */

    for (int frame = 0; frame < MEL_NUM_FRAMES; frame++) {
        for (int k = 0; k < MEL_NUM_MFCC; k++) {
            float sum = 0.0f;
            const float *dct_row = &s_dct_matrix[k * N_MELS];

            for (int m = 0; m < N_MELS; m++) {
                sum += s_log_mel[m * MEL_NUM_FRAMES + frame] * dct_row[m];
            }

            output[frame * MEL_NUM_MFCC + k] = sum;
        }
    }
}
