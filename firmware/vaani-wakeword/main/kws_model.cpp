/**
 * @file kws_model.cpp
 * @brief TFLite Micro wrapper for the Vaani V2 DS-CNN wakeword model
 *
 * Loads the INT8 quantized model embedded in model_data.cc,
 * registers operators used by the V2 architecture, and provides
 * C-callable functions for initialization and inference.
 */

#include "kws_model.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_log.h"
#include <cmath>
#include <cstring>

static const char *TAG = "KWS_MODEL";


/* ================================================================
 * Tensor arena
 * ================================================================ */

static constexpr size_t kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t s_tensor_arena[kTensorArenaSize];


/* ================================================================
 * Model / interpreter state
 * ================================================================ */

static const tflite::Model      *s_model       = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor             *s_input       = nullptr;
static TfLiteTensor             *s_output      = nullptr;

/* Quantization parameters (read from model after init). */
static float s_input_scale       = 0.0f;
static int   s_input_zero_point  = 0;
static float s_output_scale      = 0.0f;
static int   s_output_zero_point = 0;


/* ================================================================
 * Init
 * ================================================================ */

extern "C" int kws_model_init(void)
{
    ESP_LOGI(TAG, "Loading V2 model (%u bytes)...", g_vaani_model_data_len);

    /* ----------------------------------------------------------
     * 1. Load model from flash
     * ---------------------------------------------------------- */

    s_model = tflite::GetModel(g_vaani_model_data);

    if (s_model == nullptr) {
        ESP_LOGE(TAG, "GetModel() returned nullptr");
        return -1;
    }

    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %lu != runtime schema %d",
                 (unsigned long)s_model->version(),
                 TFLITE_SCHEMA_VERSION);
        return -1;
    }

    ESP_LOGI(TAG, "Model loaded (schema v%lu)", (unsigned long)s_model->version());


    /* ----------------------------------------------------------
     * 2. Register operators used by V2 DS-CNN architecture:
     *    Conv2D, DepthwiseConv2D, Mean, FullyConnected, Softmax
     * ---------------------------------------------------------- */

    static tflite::MicroMutableOpResolver<5> resolver;

    if (resolver.AddConv2D()          != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddMean()            != kTfLiteOk ||
        resolver.AddFullyConnected()  != kTfLiteOk ||
        resolver.AddSoftmax()         != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Failed to register operators");
        return -1;
    }

    ESP_LOGI(TAG, "5 operators registered");


    /* ----------------------------------------------------------
     * 3. Create interpreter
     * ---------------------------------------------------------- */

    static tflite::MicroInterpreter interpreter(
        s_model,
        resolver,
        s_tensor_arena,
        kTensorArenaSize
    );

    s_interpreter = &interpreter;


    /* ----------------------------------------------------------
     * 4. Allocate tensors
     * ---------------------------------------------------------- */

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -1;
    }

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    if (s_input == nullptr || s_output == nullptr) {
        ESP_LOGE(TAG, "Input or output tensor is null");
        return -1;
    }


    /* ----------------------------------------------------------
     * 5. Read quantization parameters
     * ---------------------------------------------------------- */

    s_input_scale       = s_input->params.scale;
    s_input_zero_point  = s_input->params.zero_point;
    s_output_scale      = s_output->params.scale;
    s_output_zero_point = s_output->params.zero_point;


    /* ----------------------------------------------------------
     * 6. Log tensor info
     * ---------------------------------------------------------- */

    ESP_LOGI(TAG, "--- INPUT TENSOR ---");
    ESP_LOGI(TAG, "  Type      : %d", s_input->type);
    ESP_LOGI(TAG, "  Shape     : %d x %d x %d x %d",
             s_input->dims->data[0], s_input->dims->data[1],
             s_input->dims->data[2], s_input->dims->data[3]);
    ESP_LOGI(TAG, "  Scale     : %.10f", s_input_scale);
    ESP_LOGI(TAG, "  Zero point: %d", s_input_zero_point);

    ESP_LOGI(TAG, "--- OUTPUT TENSOR ---");
    ESP_LOGI(TAG, "  Type      : %d", s_output->type);
    ESP_LOGI(TAG, "  Classes   : %d", s_output->dims->data[1]);
    ESP_LOGI(TAG, "  Scale     : %.10f", s_output_scale);
    ESP_LOGI(TAG, "  Zero point: %d", s_output_zero_point);

    size_t arena_used = s_interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena used: %u / %u bytes",
             (unsigned)arena_used, (unsigned)kTensorArenaSize);

    ESP_LOGI(TAG, "V2 Model ready.");
    return 0;
}


/* ================================================================
 * Run inference
 * ================================================================ */

extern "C" int kws_model_run(const float *features,
                              float *p_silence, float *p_unknown, float *p_vaani)
{
    /* ----------------------------------------------------------
     * Quantize float32 features -> INT8:
     *   quantized = round(value / scale) + zero_point
     *   clamped to [-128, 127]
     * ---------------------------------------------------------- */

    int8_t *input_data = s_input->data.int8;
    int     total      = s_input->bytes;    /* 63 * 13 = 819 */

    for (int i = 0; i < total; i++) {
        float q = roundf(features[i] / s_input_scale)
                + (float)s_input_zero_point;

        if (q < -128.0f) q = -128.0f;
        if (q >  127.0f) q =  127.0f;

        input_data[i] = (int8_t)q;
    }


    /* ----------------------------------------------------------
     * Invoke
     * ---------------------------------------------------------- */

    if (s_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return -1;
    }


    /* ----------------------------------------------------------
     * Dequantize 3 INT8 outputs -> float32 probabilities:
     *   real_value = (quantized - zero_point) * scale
     * ---------------------------------------------------------- */

    if (p_silence) {
        *p_silence = (float)(s_output->data.int8[0] - s_output_zero_point)
                   * s_output_scale;
    }

    if (p_unknown) {
        *p_unknown = (float)(s_output->data.int8[1] - s_output_zero_point)
                   * s_output_scale;
    }

    if (p_vaani) {
        *p_vaani   = (float)(s_output->data.int8[2] - s_output_zero_point)
                   * s_output_scale;
    }

    return 0;
}
