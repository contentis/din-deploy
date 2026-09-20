# SPDX-License-Identifier: Apache-2.0
enable_language(C)
include(FetchContent)
FetchContent_Declare(din_imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG 3912b3d9a9c1b3f17431aebafd86d2f40ee6e59c)
FetchContent_Declare(din_sdl
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG 7f3ae3d57459e59943a4ecfefc8f6277ec6bf540)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_OPENGL OFF CACHE BOOL "" FORCE)
set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_VULKAN ON CACHE BOOL "" FORCE)
set(SDL_VULKAN ON CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D11 OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D12 OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(din_imgui din_sdl)
find_package(Threads REQUIRED)
add_library(din_ui STATIC file_dialog.cpp files.cpp imgui_support.cpp logo.cpp
    ${din_imgui_SOURCE_DIR}/imgui.cpp ${din_imgui_SOURCE_DIR}/imgui_draw.cpp
    ${din_imgui_SOURCE_DIR}/imgui_tables.cpp ${din_imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${din_imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
    ${din_imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${din_imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp)
target_include_directories(din_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${din_imgui_SOURCE_DIR} ${din_imgui_SOURCE_DIR}/backends)
target_link_libraries(din_ui PUBLIC SDL3::SDL3 Threads::Threads)
if(WIN32)
    target_link_libraries(din_ui PRIVATE ole32 shell32 uuid)
endif()
if(MSVC)
    target_compile_options(din_ui PRIVATE /utf-8)
    target_compile_definitions(din_ui PRIVATE NOMINMAX)
endif()
