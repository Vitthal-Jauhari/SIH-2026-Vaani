/**
 * @file main_dual_core_refactor.c
 * @brief Vaani Wakeword Detection — Dual-Core Refactor (skeleton)
 *
 * NOT a drop-in replacement — this is a reference skeleton showing how to
 * split the current single-loop main.c across both ESP32 cores so that:
 *
 *   1. Audio capture + VAD (cheap, DMA-bound) runs continuously on core 0
 *      and costs ~0% CPU while idle (blocks on I2S DMA wait).
 *   2. Feature extraction + inference (expensive, CPU-bound) runs on core 1
 *      ONLY when a full, VAD-active window is ready — it blocks on a queue
 *      and costs 0% CPU between activations.
 *   3. Ping-pong buffering means capture never stalls waiting on DSP/inference.
 *   4. If inference is still busy when the next window is ready, that hop
 *      is DROPPED (not queued) — this bounds worst-case duty cycle instead
 *      of letting a backlog build up.
 *
 * IMPORTANT: 136ms feature extraction / 133ms inference (from your serial
 * log) is the real bottleneck. This refactor makes idle CPU usage near-zero,
 * but during an ACTIVE utterance, core 1 will still be busy ~270ms per hop.
 * If you need low duty cycle even during speech, profile mel_features.c —
 * a hand-written FFT/DCT loop at this size should not take >100ms on a
 * 240MHz Xtensa core. Switching to esp-dsp's hardware-optimized FFT
 * (dsps_fft2r_fc32_ae32 or similar) is very likely a 10-20x speedup and
 * should be done alongside this change, not instead of it.
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_pm.h"
#include "driver/gpio.h"

#include "i2s_mic.h"
#include "vad.h"
#include "mel_features.h"
#include "kws_model.h"
#include "wifi_manager.h"
#include "ws_streamer.h"
#include "wifi_config.h"

static const char *TAG = "VAANI";

/* ================================================================
 * Configuration
 * ================================================================ */

#define SAMPLE_RATE         16000
#define FRAME_SAMPLES       480         /* 30 ms at 16 kHz */
#define AUDIO_BUF_SAMPLES   16000       /* 1 second */
#define KWS_HOP_SAMPLES     3200        /* 200 ms at 16 kHz */
#define KWS_THRESHOLD       0.80f
#define KWS_MIN_CONSECUTIVE 2       /* Require 2 consecutive windows to confirm — filters
                                       isolated single-window spikes; does NOT filter
                                       sustained multi-window false-positive clusters
                                       (e.g. the 4-consecutive-window run seen in
                                       true_silence.wav testing) — that needs the model fix. */

#define LED_INDICATOR_GPIO  2
#define LED_HOLD_TIME_US    2000000ULL

#define CAPTURE_TASK_CORE   0            /* PRO_CPU — audio + VAD */
#define INFER_TASK_CORE     1            /* APP_CPU — DSP + inference */
#define CAPTURE_TASK_PRIO   5
#define INFER_TASK_PRIO     4            /* lower than capture: never starve mic reads */

/* ================================================================
 * Single snapshot buffer — NOT ping-pong.
 *
 * A ping-pong pair was tried first but cost an extra 32KB of static
 * DRAM (two 16000-sample buffers instead of one), which overflowed
 * the ESP32-WROOM-32's dram0_0_seg at link time. It's also
 * unnecessary: s_infer_busy already guarantees core0 never writes
 * into this buffer while core1 (still) owns it — that's exactly
 * what "drop the hop if busy" enforces. One buffer is sufficient
 * and safe.
 * ================================================================ */

static int16_t s_window_buf[AUDIO_BUF_SAMPLES];
static volatile int s_infer_busy = 0;   /* set while core1 owns the buffer */

typedef enum {
    SYSTEM_STATE_LISTENING = 0,
    SYSTEM_STATE_STREAMING,
} system_state_t;

static volatile system_state_t s_system_state = SYSTEM_STATE_LISTENING;
static volatile int64_t s_stream_start_us = 0;
static volatile bool s_need_buffer_flush = false;
static volatile int64_t s_last_feat_us = 0;
static volatile int64_t s_last_infer_us = 0;

typedef struct {
    int16_t *buf;       /* always &s_window_buf[0]; kept as a pointer so
                            the queue payload stays uniform if a second
                            buffer is ever reintroduced deliberately */
} window_msg_t;

static QueueHandle_t s_window_queue;    /* depth 1 — deliberately shallow */

/* ================================================================
 * Core 0 — Audio capture + VAD task
 * ================================================================ */

static void audio_capture_task(void *arg)
{
    static int16_t rolling_buf[AUDIO_BUF_SAMPLES];
    static int16_t frame_buf[FRAME_SAMPLES];

    memset(rolling_buf, 0, sizeof(rolling_buf));

    int total_samples_read = 0;
    int samples_since_kws  = 0;

    ESP_LOGI(TAG, "[core0] audio_capture_task started");

    while (1) {
        /* Blocking I2S DMA read — yields CPU while waiting for samples.
         * This is the reason core 0 stays near 0% CPU when there's
         * simply no new audio ready yet. */
        esp_err_t ret = i2s_mic_read(frame_buf, FRAME_SAMPLES);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "[core0] I2S read error, skipping frame");
            continue;
        }

        /* Check if audio buffer needs flush following completed stream */
        if (s_need_buffer_flush) {
            memset(rolling_buf, 0, sizeof(rolling_buf));
            total_samples_read = 0;
            samples_since_kws = 0;
            s_need_buffer_flush = false;
            ESP_LOGI(TAG, "[stream] Audio buffer flushed, KWS listening resumed.");
        }

        /* If streaming mode is active, pipe raw PCM directly over warm WebSocket */
        if (s_system_state == SYSTEM_STATE_STREAMING) {
            int64_t now_us = esp_timer_get_time();
            int64_t elapsed_us = now_us - s_stream_start_us;

            if (elapsed_us >= (int64_t)(VAANI_STREAM_MAX_SEC * 1000000ULL)) {
                ESP_LOGW(TAG, "[stream] Safety timeout (%.1fs) reached, stopping stream", (double)VAANI_STREAM_MAX_SEC);
                ws_streamer_send_text("{\"action\": \"flush\"}");
                s_system_state = SYSTEM_STATE_LISTENING;
                s_need_buffer_flush = true;
                gpio_set_level(LED_INDICATOR_GPIO, 0);
            } else {
                ws_streamer_send_pcm(frame_buf, FRAME_SAMPLES * sizeof(int16_t));
            }
            continue; /* Skip KWS inference window while streaming command */
        }

        memmove(rolling_buf, rolling_buf + FRAME_SAMPLES,
                (AUDIO_BUF_SAMPLES - FRAME_SAMPLES) * sizeof(int16_t));
        memcpy(rolling_buf + (AUDIO_BUF_SAMPLES - FRAME_SAMPLES),
               frame_buf, FRAME_SAMPLES * sizeof(int16_t));

        total_samples_read += FRAME_SAMPLES;

        vad_process_frame(frame_buf, FRAME_SAMPLES);

        samples_since_kws += FRAME_SAMPLES;
        if (samples_since_kws < KWS_HOP_SAMPLES) {
            continue;
        }
        samples_since_kws = 0;

        if (total_samples_read < AUDIO_BUF_SAMPLES) {
            continue;   /* still filling the first window */
        }

        if (!vad_is_active()) {
            continue;   /* cheap gate — most hops end here, near-zero cost */
        }

        /* Don't hand off a new window if core1 hasn't finished the last
         * one — DROP this hop rather than let a backlog form. This is
         * what actually bounds worst-case duty cycle. */
        if (s_infer_busy) {
            ESP_LOGW(TAG, "[core0] inference still busy, dropping hop");
            continue;
        }

        /* Copy into the snapshot buffer and hand it off. Safe because
         * s_infer_busy is only cleared once core1 has finished reading
         * this buffer — see the gate above. */
        memcpy(s_window_buf, rolling_buf,
               AUDIO_BUF_SAMPLES * sizeof(int16_t));

        window_msg_t msg = { .buf = s_window_buf };
        s_infer_busy = 1;

        if (xQueueSend(s_window_queue, &msg, 0) != pdTRUE) {
            /* Queue depth is 1 by design — should never happen given the
             * s_infer_busy gate above, but fail safe. */
            s_infer_busy = 0;
            ESP_LOGW(TAG, "[core0] queue send failed unexpectedly");
        }
    }
}

/* ================================================================
 * Core 1 — Feature extraction + inference task
 * ================================================================ */

static void kws_infer_task(void *arg)
{
    static float features[TOTAL_FEATURES];
    int total_detections = 0;
    int64_t led_off_time = 0;
    int s_consecutive_hits = 0;

    ESP_LOGI(TAG, "[core1] kws_infer_task started");

    while (1) {
        window_msg_t msg;

        /* Blocks here indefinitely — costs ~0% CPU between activations.
         * This is the core mechanism that gets idle CPU usage down. */
        if (xQueueReceive(s_window_queue, &msg, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (led_off_time > 0 && esp_timer_get_time() >= led_off_time) {
            if (s_system_state != SYSTEM_STATE_STREAMING) {
                gpio_set_level(LED_INDICATOR_GPIO, 0);
                led_off_time = 0;
            }
        }

        int64_t t0 = esp_timer_get_time();
        mel_features_extract(msg.buf, AUDIO_BUF_SAMPLES, features);
        int64_t t_feat = esp_timer_get_time() - t0;

        float p_silence = 0.0f, p_unknown = 0.0f, p_vaani = 0.0f;

        int64_t t1 = esp_timer_get_time();
        int rc = kws_model_run(features, &p_silence, &p_unknown, &p_vaani);
        int64_t t_infer = esp_timer_get_time() - t1;

        s_infer_busy = 0;   /* release the buffer back to core0 */

        if (rc != 0) {
            ESP_LOGE(TAG, "[core1] Inference failed!");
            continue;
        }

        s_last_feat_us = t_feat;
        s_last_infer_us = t_infer;

        bool raw_hit = (p_vaani >= KWS_THRESHOLD);

        if (raw_hit) {
            s_consecutive_hits++;
            if (s_consecutive_hits == KWS_MIN_CONSECUTIVE) {
                total_detections++;
                gpio_set_level(LED_INDICATOR_GPIO, 1);

                int64_t t_wake = esp_timer_get_time();
                int64_t handoff_us = 0;

                if (ws_streamer_is_connected()) {
                    s_stream_start_us = esp_timer_get_time();
                    s_system_state = SYSTEM_STATE_STREAMING;
                    handoff_us = esp_timer_get_time() - t_wake;
                    printf("\n🔔 [WAKE] \"Vaani\" CONFIRMED (P=%.4f) | handoff=%.2f ms | Streaming to ASR Server...\n",
                           p_vaani, (float)handoff_us / 1000.0f);
                } else {
                    printf("\n🔔 [WAKE] \"Vaani\" CONFIRMED (P=%.4f) | WebSocket not connected!\n", p_vaani);
                    led_off_time = esp_timer_get_time() + LED_HOLD_TIME_US;
                }
            }
        } else {
            s_consecutive_hits = 0;
            /* Only log candidate if confidence is notable to avoid flooding console */
            if (p_vaani >= 0.40f) {
                printf("  [kws] Candidate: P(Vaani)=%.4f  P(Unknown)=%.4f\n", p_vaani, p_unknown);
            }
        }
    }
}

/* ================================================================
 * Real-Time Telemetry & Resource Profiling Task
 * ================================================================ */

static void telemetry_task(void *arg)
{
    (void)arg;
    static TaskStatus_t prev_tasks[24];
    static UBaseType_t prev_count = 0;
    static uint32_t prev_total_runtime = 0;

    vTaskDelay(pdMS_TO_TICKS(3000)); /* Allow boot sequence and Wi-Fi to settle */

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(2000));

        TaskStatus_t curr_tasks[24];
        uint32_t curr_total_runtime = 0;
        UBaseType_t curr_count = uxTaskGetSystemState(curr_tasks, 24, &curr_total_runtime);

        uint32_t free_heap = esp_get_free_heap_size();
        uint32_t min_free_heap = esp_get_minimum_free_heap_size();

        if (curr_count > 0 && prev_count > 0 && curr_total_runtime > prev_total_runtime) {
            uint32_t total_delta = curr_total_runtime - prev_total_runtime;
            uint32_t idle0_delta = 0;
            uint32_t idle1_delta = 0;

            for (int i = 0; i < curr_count; i++) {
                for (int j = 0; j < prev_count; j++) {
                    if (curr_tasks[i].xHandle == prev_tasks[j].xHandle) {
                        uint32_t dt = curr_tasks[i].ulRunTimeCounter - prev_tasks[j].ulRunTimeCounter;
                        if (strcmp(curr_tasks[i].pcTaskName, "IDLE0") == 0) {
                            idle0_delta = dt;
                        } else if (strcmp(curr_tasks[i].pcTaskName, "IDLE1") == 0) {
                            idle1_delta = dt;
                        }
                        break;
                    }
                }
            }

            float core0_idle = (float)idle0_delta * 100.0f / (float)total_delta;
            float core1_idle = (float)idle1_delta * 100.0f / (float)total_delta;
            if (core0_idle > 100.0f) core0_idle = 100.0f;
            if (core1_idle > 100.0f) core1_idle = 100.0f;
            float total_idle = (core0_idle + core1_idle) / 2.0f;

            const char *state_str = (s_system_state == SYSTEM_STATE_STREAMING) ? "STREAMING" : "LISTENING";

            printf("[TELEMETRY] state=%s cpu_idle=%.1f%% core0_idle=%.1f%% core1_idle=%.1f%% heap_free_kb=%.1f heap_min_kb=%.1f feat_ms=%.1f infer_ms=%.1f\n",
                   state_str,
                   total_idle,
                   core0_idle,
                   core1_idle,
                   (float)free_heap / 1024.0f,
                   (float)min_free_heap / 1024.0f,
                   (float)s_last_feat_us / 1000.0f,
                   (float)s_last_infer_us / 1000.0f);
        }

        if (curr_count > 0) {
            memcpy(prev_tasks, curr_tasks, sizeof(TaskStatus_t) * curr_count);
            prev_count = curr_count;
            prev_total_runtime = curr_total_runtime;
        }
    }
}

/* ================================================================
 * Phase 1 Boot Test: Background Wi-Fi & Warm WebSocket Task
 * ================================================================ */

static void ws_text_rx_cb(const char *data, size_t len)
{
    /* Check if this is a transcription response */
    if (strstr(data, "\"type\": \"transcription\"") != NULL ||
        strstr(data, "\"type\":\"transcription\"") != NULL) {

        printf("\n🎯 [ASR] %.*s\n\n", (int)len, data);

        /* Stop streaming and return to listening */
        if (s_system_state == SYSTEM_STATE_STREAMING) {
            s_system_state = SYSTEM_STATE_LISTENING;
            s_need_buffer_flush = true;
            gpio_set_level(LED_INDICATOR_GPIO, 0);
        }
    }
}

static void wifi_ws_boot_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "[net] Initializing Wi-Fi Station...");
    esp_err_t err = wifi_manager_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[net] Wi-Fi init failed: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "[net] Waiting for Wi-Fi IP assignment...");
    while (!wifi_manager_is_connected()) {
        vTaskDelay(pdMS_TO_TICKS(200));
    }

    ESP_LOGI(TAG, "[net] Wi-Fi connected! Connecting warm WebSocket to: %s", VAANI_ASR_WS_URL);
    err = ws_streamer_init(VAANI_ASR_WS_URL, ws_text_rx_cb);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[net] Failed to init WebSocket: %s", esp_err_to_name(err));
    }

    vTaskDelete(NULL);
}

/* ================================================================
 * app_main — init subsystems, create pinned tasks, done.
 * ================================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "==== VAANI WAKEWORD DETECTION v2 (dual-core) ====");

    /* Optional: enable automatic light-sleep so core 0 actually clocks
     * down between DMA interrupts instead of just spinning at full
     * frequency waiting. Requires CONFIG_PM_ENABLE +
     * CONFIG_FREERTOS_USE_TICKLESS_IDLE in sdkconfig. Tune min/max freq
     * to your board's stability envelope before relying on this for
     * the idle-power budget. */
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 240,
        .min_freq_mhz = 80,
        .light_sleep_enable = true,
    };
    esp_err_t pm_ret = esp_pm_configure(&pm_config);
    if (pm_ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_pm_configure failed (%d) — check sdkconfig PM options", pm_ret);
    }

    ESP_ERROR_CHECK(i2s_mic_init());
    mel_features_init();
    if (kws_model_init() != 0) {
        ESP_LOGE(TAG, "Model init failed! Halting.");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }
    vad_init();

    gpio_config_t led_cfg = {
        .pin_bit_mask = (1ULL << LED_INDICATOR_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&led_cfg);
    gpio_set_level(LED_INDICATOR_GPIO, 0);

    s_window_queue = xQueueCreate(1, sizeof(window_msg_t));
    if (s_window_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create window queue! Halting.");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture",
                             4096, NULL, CAPTURE_TASK_PRIO, NULL,
                             CAPTURE_TASK_CORE);

    xTaskCreatePinnedToCore(kws_infer_task, "kws_infer",
                             8192, NULL, INFER_TASK_PRIO, NULL,
                             INFER_TASK_CORE);

    ESP_LOGI(TAG, "Tasks launched: capture on core %d, inference on core %d",
             CAPTURE_TASK_CORE, INFER_TASK_CORE);

    /* Launch background network connection task */
    xTaskCreate(wifi_ws_boot_task, "wifi_ws_boot", 4096, NULL, 3, NULL);

    /* Launch background telemetry task for real-time CPU, RAM & latency profiling */
    xTaskCreate(telemetry_task, "telemetry", 4096, NULL, 1, NULL);

    /* app_main can return here — all tasks keep running independently. */
}