# Qwen3 offline ASR: processing library plus a thin CLI, as in Whisper.
add_din_shared_library(din_asr_qwen3 STATIC qwen3.cpp)
target_include_directories(din_asr_qwen3 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(din_asr_qwen3 PUBLIC din_common_ort din_common_io nlohmann_json::nlohmann_json
    PRIVATE CUDA::cudart)
add_din_executable(din_asr_qwen3_cli main.cpp)
target_link_libraries(din_asr_qwen3_cli PRIVATE din_asr_qwen3 argparse)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
