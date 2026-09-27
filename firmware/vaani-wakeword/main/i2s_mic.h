/**
 * @file i2s_mic.h
 * @brief I2S INMP441 MEMS microphone driver for ESP32
 *
 * Configures the I2S peripheral in RX-only standard (Philips) mode
 * for capturing 16 kHz mono audio from an INMP441 microphone.
 *
 * Default GPIO mapping (change in i2s_mic.c if needed):
 *   BCK  → GPIO 26
 *   WS   → GPIO 25
 *   DIN  → GPIO 22
 */

#ifndef I2S_MIC_H_
#define I2S_MIC_H_

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

/**
 * @brief Initialize the I2S peripheral for microphone input.
 *
 * Must be called once before any calls to i2s_mic_read().
 *
 * @return ESP_OK on success, or an error code.
 */
esp_err_t i2s_mic_init(void);

/**
 * @brief Read audio samples from the microphone.
 *
 * Blocks until the requested number of samples is available.
 * Internally reads 32-bit I2S data and converts to 16-bit PCM.
 *
 * @param out_buf     Output buffer for 16-bit signed PCM samples.
 * @param num_samples Number of samples to read.
 * @return ESP_OK on success, or an error code.
 */
esp_err_t i2s_mic_read(int16_t *out_buf, size_t num_samples);

#endif  /* I2S_MIC_H_ */
