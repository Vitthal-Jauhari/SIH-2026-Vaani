idf_component_register(
    SRCS "main.c" "i2s_mic.c" "vad.c" "mel_features.c" "kws_model.cpp" "model_data.cc"
         "wifi_manager.c" "ws_streamer.c"
    INCLUDE_DIRS "."
)
