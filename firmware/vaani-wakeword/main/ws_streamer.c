#include "ws_streamer.h"

#include <string.h>
#include "esp_log.h"
#include "esp_websocket_client.h"

static const char *TAG = "VAANI_WS";

static esp_websocket_client_handle_t s_client = NULL;
static ws_rx_cb_t s_rx_callback = NULL;
static volatile bool s_connected = false;

static void websocket_event_handler(
    void *handler_args,
    esp_event_base_t base,
    int32_t event_id,
    void *event_data)
{
    (void)handler_args;
    (void)base;
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;

    switch ((esp_websocket_event_id_t)event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
        s_connected = true;
        ESP_LOGI(TAG, "WebSocket connected");
        break;

    case WEBSOCKET_EVENT_DISCONNECTED:
        s_connected = false;
        ESP_LOGW(TAG, "WebSocket disconnected");
        break;

    case WEBSOCKET_EVENT_DATA:
        /*
         * Only forward text frames to the application callback.
         * Binary frames from the server are ignored for now.
         */
        if (data != NULL && data->op_code == 0x1 && data->data_ptr != NULL && data->data_len > 0) {
            if (s_rx_callback != NULL) {
                s_rx_callback(data->data_ptr, (size_t)data->data_len);
            }
        }
        break;

    case WEBSOCKET_EVENT_ERROR:
        s_connected = false;
        ESP_LOGE(TAG, "WebSocket error");
        break;

    case WEBSOCKET_EVENT_CLOSED:
        s_connected = false;
        ESP_LOGW(TAG, "WebSocket closed");
        break;

    default:
        break;
    }
}

esp_err_t ws_streamer_init(const char *url, ws_rx_cb_t rx_callback)
{
    if (url == NULL || url[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_client != NULL) {
        ESP_LOGW(TAG, "WebSocket client already initialized");
        return ESP_ERR_INVALID_STATE;
    }

    s_rx_callback = rx_callback;

    esp_websocket_client_config_t config = {
        .uri = url,
    };

    s_client = esp_websocket_client_init(&config);
    if (s_client == NULL) {
        ESP_LOGE(TAG, "Failed to create WebSocket client");
        return ESP_FAIL;
    }

    esp_err_t err = esp_websocket_register_events(
        s_client, WEBSOCKET_EVENT_ANY, websocket_event_handler, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register WebSocket events: %s", esp_err_to_name(err));
        esp_websocket_client_destroy(s_client);
        s_client = NULL;
        return err;
    }

    ESP_LOGI(TAG, "Starting warm WebSocket: %s", url);
    err = esp_websocket_client_start(s_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start WebSocket: %s", esp_err_to_name(err));
        esp_websocket_client_destroy(s_client);
        s_client = NULL;
        return err;
    }

    return ESP_OK;
}

bool ws_streamer_is_connected(void)
{
    return s_connected && s_client != NULL && esp_websocket_client_is_connected(s_client);
}

esp_err_t ws_streamer_send_pcm(const void *data, size_t len)
{
    if (data == NULL || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!ws_streamer_is_connected()) {
        return ESP_ERR_INVALID_STATE;
    }

    int sent = esp_websocket_client_send_bin(
        s_client, (const char *)data, (int)len, pdMS_TO_TICKS(100));
    if (sent < 0) {
        ESP_LOGW(TAG, "PCM send failed");
        return ESP_FAIL;
    }
    if ((size_t)sent != len) {
        ESP_LOGW(TAG, "Partial PCM send: %d/%u bytes", sent, (unsigned)len);
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t ws_streamer_send_text(const char *text)
{
    if (text == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!ws_streamer_is_connected()) {
        return ESP_ERR_INVALID_STATE;
    }

    int len = (int)strlen(text);
    int sent = esp_websocket_client_send_text(
        s_client, text, len, pdMS_TO_TICKS(100));
    if (sent < 0 || sent != len) {
        ESP_LOGW(TAG, "Text send failed");
        return ESP_FAIL;
    }

    return ESP_OK;
}
