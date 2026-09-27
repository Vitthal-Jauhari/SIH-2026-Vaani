/**
 * @file vad.h
 * @brief Energy-based Voice Activity Detection (VAD)
 *
 * Simple RMS energy VAD gate ported from the Python test runner
 * (test_runners/model_v1/realtime_vad_test.py).
 *
 * Uses a state machine with configurable speech-start and
 * speech-end frame counters to debounce noisy transitions.
 */

#ifndef VAD_H_
#define VAD_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/**
 * @brief Initialize / reset the VAD state machine.
 */
void vad_init(void);

/**
 * @brief Process one audio frame and update VAD state.
 *
 * @param frame     Pointer to 16-bit PCM samples.
 * @param num_samples Number of samples in this frame.
 * @return Measured RMS energy in dBFS for this frame.
 */
float vad_process_frame(const int16_t *frame, size_t num_samples);

/**
 * @brief Check whether speech is currently active.
 *
 * @return true if VAD considers speech active, false otherwise.
 */
bool vad_is_active(void);

#endif  /* VAD_H_ */
