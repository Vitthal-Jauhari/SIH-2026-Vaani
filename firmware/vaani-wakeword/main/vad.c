/**
 * @file vad.c
 * @brief Energy-based Voice Activity Detection
 *
 * Ported from test_runners/model_v1/realtime_vad_test.py.
 *
 * Computes RMS energy in dBFS for each audio frame and uses a
 * simple state machine with speech/silence frame counters to
 * debounce transitions.
 *
 * Configuration:
 *   Threshold  : -45 dBFS  (adjust for your environment)
 *   Start count:  2 consecutive speech frames to activate
 *   End count  :  5 consecutive silent frames to deactivate
 */

#include "vad.h"

#include <math.h>
#include "esp_log.h"

static const char *TAG = "VAD";


/* ================================================================
 * Configuration  —  tune these for your environment
 * ================================================================ */

#define VAD_THRESHOLD_DB    (-34.0f)
#define VAD_START_FRAMES    2
#define VAD_END_FRAMES      20          /* 20 frames = 600 ms hangover to cover 1s buffer */


/* ================================================================
 * State
 * ================================================================ */

static int  s_speech_frames = 0;
static int  s_silent_frames = 0;
static bool s_active        = false;


/* ================================================================
 * Init
 * ================================================================ */

void vad_init(void)
{
    s_speech_frames = 0;
    s_silent_frames = 0;
    s_active        = false;

    ESP_LOGI(TAG, "VAD initialized (threshold %.1f dBFS, "
             "start %d frames, end %d frames)",
             VAD_THRESHOLD_DB, VAD_START_FRAMES, VAD_END_FRAMES);
}


/* ================================================================
 * Process one frame
 * ================================================================ */

float vad_process_frame(const int16_t *frame, size_t num_samples)
{
    /* ----------------------------------------------------------
     * Compute RMS energy in dBFS.
     *
     * INT16 audio: full scale = 32768.
     * dBFS = 20 * log10(rms / 32768)
     * ---------------------------------------------------------- */

    float sum_sq = 0.0f;

    for (size_t i = 0; i < num_samples; i++) {
        float s = (float)frame[i];
        sum_sq += s * s;
    }

    float rms = sqrtf(sum_sq / (float)num_samples);

    float db;
    if (rms < 1.0f) {
        db = -100.0f;
    } else {
        db = 20.0f * log10f(rms / 32768.0f);
    }


    /* ----------------------------------------------------------
     * Update speech / silence counters
     * ---------------------------------------------------------- */

    bool frame_is_speech = (db >= VAD_THRESHOLD_DB);

    if (frame_is_speech) {
        s_speech_frames++;
        s_silent_frames = 0;
    } else {
        s_silent_frames++;
        s_speech_frames = 0;
    }


    /* ----------------------------------------------------------
     * State transitions
     * ---------------------------------------------------------- */

    /* Activate after consecutive speech frames */
    if (!s_active && s_speech_frames >= VAD_START_FRAMES) {
        s_active = true;
        ESP_LOGI(TAG, "Speech START (energy: %.1f dBFS >= threshold %.1f)", db, VAD_THRESHOLD_DB);
    }

    /* Deactivate after consecutive silent frames */
    if (s_active && s_silent_frames >= VAD_END_FRAMES) {
        s_active = false;
        ESP_LOGI(TAG, "Speech END (energy: %.1f dBFS < threshold %.1f)", db, VAD_THRESHOLD_DB);
    }

    return db;
}


/* ================================================================
 * Query
 * ================================================================ */

bool vad_is_active(void)
{
    return s_active;
}
