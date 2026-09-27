#pragma once

/*
 * Example development configuration.
 *
 * Copy this file to wifi_config.h and update with your actual credentials:
 * cp wifi_config.example.h wifi_config.h
 */
#define VAANI_WIFI_SSID      "YOUR_WIFI_SSID"
#define VAANI_WIFI_PASS      "YOUR_WIFI_PASSWORD"

/*
 * Use your laptop's local LAN IPv4 address (e.g., from ipconfig).
 *
 * Example:
 * ws://192.168.1.105:8000/ws/transcribe
 */
#define VAANI_ASR_WS_URL     "ws://YOUR_LAPTOP_IP:8000/ws/transcribe"
#define VAANI_STREAM_MAX_SEC  7.0f
