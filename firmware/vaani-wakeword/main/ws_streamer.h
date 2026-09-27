#pragma once

#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

typedef void (*ws_rx_cb_t)(const char *data, size_t len);

/**
 * Initialize the persistent WebSocket client.
 *
 * The client is created and started immediately.
 * It will reconnect automatically when possible.
 */
esp_err_t ws_streamer_init(const char *url, ws_rx_cb_t rx_callback);

/**
 * Returns true only when the WebSocket handshake
 * has completed and the connection is usable.
 */
bool ws_streamer_is_connected(void);

/**
 * Send one binary PCM frame.
 *
 * Audio format:
 * 16 kHz
 * 16-bit signed PCM
 * mono
 */
esp_err_t ws_streamer_send_pcm(const void *data, size_t len);

/**
 * Send a text message.
 *
 * Used later for protocol control messages such as:
 * {"type":"end_of_utterance"}
 */
esp_err_t ws_streamer_send_text(const char *text);
