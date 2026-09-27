/**
 * @file kws_model.h
 * @brief TFLite Micro wrapper for the Vaani V2 DS-CNN wakeword model
 *
 * Loads the INT8 quantized model, registers the required operators,
 * and provides a C-callable interface for running inference.
 */

#ifndef KWS_MODEL_H_
#define KWS_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load the model, register ops, create interpreter, allocate tensors.
 *
 * Must be called once at startup. Logs input/output tensor metadata.
 *
 * @return 0 on success, -1 on failure.
 */
int kws_model_init(void);

/**
 * @brief Run wakeword inference on extracted 63x13 MFCC features.
 *
 * Quantizes float32 features to INT8 using the model's input
 * scale/zero-point, invokes the interpreter, and dequantizes
 * the 3 output class probabilities.
 *
 * @param features   Pointer to 63 × 13 = 819 float32 MFCC features.
 *                   Layout: [frame][mfcc], matching (1, 63, 13, 1) tensor.
 * @param p_silence  Output: probability of class 0 (Silence).
 * @param p_unknown  Output: probability of class 1 (Unknown speech / Negative).
 * @param p_vaani    Output: probability of class 2 (Vaani wakeword detected).
 * @return 0 on success, -1 on inference failure.
 */
int kws_model_run(const float *features, float *p_silence, float *p_unknown, float *p_vaani);

#ifdef __cplusplus
}
#endif

#endif  /* KWS_MODEL_H_ */
