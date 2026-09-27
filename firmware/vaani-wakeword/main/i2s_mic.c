/**
 * @file i2s_mic.c
 * @brief I2S INMP441 MEMS microphone driver (ESP-IDF v5+/v6 new API)
 *
 * Captures 16 kHz mono audio from an INMP441 via the I2S standard
 * (Philips) interface. Reads 32-bit frames and converts to 16-bit
 * PCM by taking the upper 16 bits of the 24-bit INMP441 output.
 *
 * GPIO defaults (change the #defines below to match your wiring):
 *   BCK  → GPIO 26
 *   WS   → GPIO 25
 *   DIN  → GPIO 22
 */

#include "i2s_mic.h"

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "driver/i2s_std.h"
#include "esp_log.h"

static const char *TAG = "I2S_MIC";


/* ================================================================
 * GPIO Configuration  —  CHANGE THESE TO MATCH YOUR WIRING
 * ================================================================ */

#define I2S_MIC_BCK_GPIO    26
#define I2S_MIC_WS_GPIO     25
#define I2S_MIC_DIN_GPIO    22


/* ================================================================
 * Constants
 * ================================================================ */

#define I2S_SAMPLE_RATE     16000
#define I2S_PORT            I2S_NUM_0

/* Maximum samples per single read call.
 * 480 samples = 30 ms at 16 kHz. */
#define MAX_READ_SAMPLES    480

/* Static buffer for raw 32-bit I2S data. */
static int32_t s_raw_buf[MAX_READ_SAMPLES];

/* I2S channel handle (new driver API). */
static i2s_chan_handle_t s_rx_handle = NULL;


/* ================================================================
 * Initialization
 * ================================================================ */

esp_err_t i2s_mic_init(void)
{
    ESP_LOGI(TAG, "Initializing I2S microphone...");
    ESP_LOGI(TAG, "  BCK  = GPIO %d", I2S_MIC_BCK_GPIO);
    ESP_LOGI(TAG, "  WS   = GPIO %d", I2S_MIC_WS_GPIO);
    ESP_LOGI(TAG, "  DIN  = GPIO %d", I2S_MIC_DIN_GPIO);
    ESP_LOGI(TAG, "  Rate = %d Hz", I2S_SAMPLE_RATE);

    /* ----------------------------------------------------------
     * 1. Create a new I2S channel (RX only)
     * ---------------------------------------------------------- */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_PORT,
        I2S_ROLE_MASTER
    );

    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* ----------------------------------------------------------
     * 2. Configure standard (Philips) mode
     *
     *    INMP441 outputs 24-bit audio in a 32-bit I2S frame.
     *    We read 32-bit and extract the upper 16 bits in
     *    i2s_mic_read().
     *
     *    If your INMP441 L/R pin is HIGH, change slot_mask to
     *    I2S_STD_SLOT_RIGHT.
     * ---------------------------------------------------------- */
    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_BCK_GPIO,
            .ws   = I2S_MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_MIC_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    /* Select left channel (INMP441 L/R pin = LOW or GND). */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ret = i2s_channel_init_std_mode(s_rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %s",
                 esp_err_to_name(ret));
        return ret;
    }

    /* ----------------------------------------------------------
     * 3. Enable the channel
     * ---------------------------------------------------------- */
    ret = i2s_channel_enable(s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %s",
                 esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "I2S microphone ready.");
    return ESP_OK;
}


/* ================================================================
 * Read samples
 * ================================================================ */

esp_err_t i2s_mic_read(int16_t *out_buf, size_t num_samples)
{
    if (num_samples > MAX_READ_SAMPLES) {
        ESP_LOGE(TAG, "Requested %u samples exceeds max %d",
                 (unsigned)num_samples, MAX_READ_SAMPLES);
        return ESP_ERR_INVALID_ARG;
    }

    size_t bytes_to_read = num_samples * sizeof(int32_t);
    size_t bytes_read    = 0;

    esp_err_t ret = i2s_channel_read(
        s_rx_handle,
        s_raw_buf,
        bytes_to_read,
        &bytes_read,
        pdMS_TO_TICKS(1000)
    );

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_read failed: %s",
                 esp_err_to_name(ret));
        return ret;
    }

    /* Convert 32-bit → 16-bit.
     *
     * INMP441 places 24-bit audio data left-aligned in a 32-bit
     * word (bits [31:8], bits [7:0] are zero).  Right-shifting by
     * 16 gives us the most-significant 16 bits — plenty for
     * speech processing. */

    size_t samples_read = bytes_read / sizeof(int32_t);

    for (size_t i = 0; i < samples_read; i++) {
        int32_t s = s_raw_buf[i] >> 16;
        if (s > 32767)  s = 32767;
        if (s < -32768) s = -32768;
        out_buf[i] = (int16_t)s;
    }

    /* Zero-fill if we got fewer samples than requested. */
    for (size_t i = samples_read; i < num_samples; i++) {
        out_buf[i] = 0;
    }

    return ESP_OK;
}
