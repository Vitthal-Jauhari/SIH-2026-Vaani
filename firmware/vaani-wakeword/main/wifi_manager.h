#pragma once

#include <stdbool.h>
#include "esp_err.h"

/**
 * Initialize Wi-Fi in station mode and begin connection.
 *
 * Returns ESP_OK once the Wi-Fi subsystem has been initialized.
 * Actual connection happens asynchronously.
 */
esp_err_t wifi_manager_init(void);

/**
 * Returns true when the ESP32 has obtained an IP address.
 */
bool wifi_manager_is_connected(void);
