/**
 * @file mel_features.h
 * @brief On-device MFCC feature extraction for V2 DS-CNN Wakeword Model
 *
 * Replicates the 63x13 MFCC feature pipeline on ESP32:
 *   16 kHz mono (1 second) -> 512-pt FFT (center=True, hop=256, 63 frames)
 *   -> 40 mel bins -> log dB (ref=1.0, top_db=80.0) -> 13 DCT-II (ortho)
 *
 * Output shape: 63 × 13 (time_frames × mfcc_coefficients)
 * Matches the V2 model input shape: (1, 63, 13, 1).
 */

#ifndef MEL_FEATURES_H_
#define MEL_FEATURES_H_

#include <stdint.h>
#include <stddef.h>

#define MEL_NUM_FRAMES  63
#define MEL_NUM_MFCC    13
#define TOTAL_FEATURES  (MEL_NUM_FRAMES * MEL_NUM_MFCC) /* 819 */

/**
 * @brief Pre-compute and initialize tables for feature extraction.
 *
 * Must be called once at startup before mel_features_extract().
 */
void mel_features_init(void);

/**
 * @brief Extract 63x13 MFCC features from a 1-second audio buffer.
 *
 * @param audio       Pointer to 16-bit PCM samples (16 kHz, mono).
 * @param num_samples Number of samples (should be 16000 for 1 second).
 * @param output      Output buffer holding TOTAL_FEATURES floats (819 elements).
 *                    Layout: [frame][mfcc], matching (1, 63, 13, 1) tensor.
 */
void mel_features_extract(const int16_t *audio, size_t num_samples, float *output);

#endif  /* MEL_FEATURES_H_ */
