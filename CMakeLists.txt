cmake_minimum_required(VERSION 3.16)
if(EXISTS "${CMAKE_SOURCE_DIR}/local_override.cmake")
    include("${CMAKE_SOURCE_DIR}/local_override.cmake")
    return()
endif()
include(local.cmake OPTIONAL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard set to ${CMAKE_CXX_STANDARD}")
project(swaptube LANGUAGES CXX)
message(STATUS "Project '${PROJECT_NAME}' initialized")

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libswscale libavcodec libavformat libavdevice libavutil libavfilter)
pkg_check_modules(GLIB REQUIRED glib-2.0)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)

# Cairo, RSVG and gdk-pixbuf are discovered through pkg-config rather than
# hardcoded /usr/include paths, because their headers include each other's and
# only pkg-config knows the transitive include directories. On a Homebrew
# prefix none of them live anywhere the compiler looks by default.
pkg_check_modules(CAIRO REQUIRED cairo)
pkg_check_modules(RSVG REQUIRED librsvg-2.0)
pkg_check_modules(GDKPIXBUF REQUIRED gdk-pixbuf-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
pkg_check_modules(LZMA liblzma)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

# Add globbed cpp files: all .cpp files in src, including subdirectories.
# Exclude the src/Projects/ directory.
# Include, however, src/Projects/.active_project.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/Projects/.*")
file(GLOB_RECURSE ACTIVE_PROJECT_SOURCE "src/Projects/.active_project.cpp")

include(CheckLanguage)
message(STATUS "Checking for CUDA and HIP support...")
check_language(CUDA)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    check_language(HIP)
endif()

# Check HIP compiler first, as it's the more specific option
# More likely to have a CUDA compiler with only HIP compute capability than vice versa
if(CMAKE_HIP_COMPILER AND NOT COMPUTE_LANG OR COMPUTE_LANG STREQUAL "HIP")
    message(STATUS "HIP compiler detected.")
    enable_language(HIP)
    add_compile_definitions(USE_AMD)
    message(STATUS "HIP compiler found: ${CMAKE_HIP_COMPILER}")
    set(GPU_LANGUAGE HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_ARCHITECTURES native)
    find_program(HIPIFY_PERL hipify-perl REQUIRED)
    file(GLOB_RECURSE CUDA_TO_TRANSLATE CONFIGURE_DEPENDS "src/CUDA/*.cu" "src/CUDA/*.cuh")
    foreach(cuda_file ${CUDA_TO_TRANSLATE})
        file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}/src/CUDA" "${cuda_file}")
        set(hip_file "${CMAKE_SOURCE_DIR}/src/HIP/${rel_path}")
        add_custom_command(
            OUTPUT ${hip_file}
            COMMAND echo "Hipifying ${rel_path}"
            COMMAND ${HIPIFY_PERL} ${cuda_file} > ${hip_file}
            DEPENDS ${cuda_file}
            VERBATIM
        )
        if(${hip_file} MATCHES ".*\.cuh$")
            list(APPEND HIP_HEADERS ${hip_file})
        elseif(${hip_file} MATCHES ".*\.cu$")
            list(APPEND GPU_SOURCES ${hip_file})
        endif()
    endforeach()
    add_custom_target(convert_headers ALL DEPENDS ${HIP_HEADERS})
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    set(CMAKE_CONFIGURE_DEPENDS ${GPU_SOURCES})
    # TODO: Check error ouput on CUDA/HIP function calls so we don't have to suppress warnings
    add_compile_options("-Wno-unused-value")
    # for parity with CUDA default behavior
    add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    add_link_options($<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
elseif(CMAKE_CUDA_COMPILER AND NOT COMPUTE_LANG OR COMPUTE_LANG STREQUAL "CUDA")
    message(STATUS "CUDA compiler detected.")
    enable_language(CUDA)
    add_compile_definitions(USE_NVIDIA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    set(GPU_LANGUAGE CUDA)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set(CMAKE_CUDA_STANDARD 17)
    else()
        set(CMAKE_CUDA_STANDARD 14)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18" AND (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native"))
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            message(WARNING "CMAKE_CUDA_ARCHITECTURES not set or set to native, but host GPU detection unsupported by CMake version. Please set a value or update to CMake >= 3.24.")
        endif()
    endif()
    file(GLOB_RECURSE GPU_SOURCES "src/CUDA/*.cu")
    set(CMAKE_CONFIGURE_DEPENDS ${GPU_SOURCES})
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe> $<$<COMPILE_LANGUAGE:CUDA>:"--diag_suppress=20012">)
    if(CMAKE_VERSION VERSION_LESS "3.17")
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>)
    endif()
else()
    # CPU backend. src/CUDA stays the source of truth; cmake/translate_cuda_to_cpu.py
    # generates src/CPU from it and rewrites the <<<grid, block>>> launches, the
    # same arrangement the HIP path has with hipify-perl. Everything else nvcc
    # would provide implicitly comes from src/CPUBackend/cuda_runtime.h.
    message(STATUS "Neither CUDA nor HIP found. Building the CPU backend.")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_compile_definitions(USE_CPU)
    set(GPU_LANGUAGE CPU)

    file(GLOB_RECURSE CUDA_TO_TRANSLATE "src/CUDA/*.cu" "src/CUDA/*.cuh")
    # Generated at configure time so the glob below sees the result; touching any
    # .cu re-runs configure and regenerates.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/translate_cuda_to_cpu.py
                --input ${CMAKE_SOURCE_DIR}/src/CUDA
                --output ${CMAKE_SOURCE_DIR}/src/CPU
        RESULT_VARIABLE TRANSLATE_RESULT
        OUTPUT_VARIABLE TRANSLATE_OUTPUT
    )
    if(NOT TRANSLATE_RESULT EQUAL 0)
        message(FATAL_ERROR "translate_cuda_to_cpu.py failed: ${TRANSLATE_OUTPUT}")
    endif()
    message(STATUS "${TRANSLATE_OUTPUT}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CUDA_TO_TRANSLATE})

    file(GLOB_RECURSE GPU_SOURCES "src/CPU/*.cu")
    set_source_files_properties(${GPU_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;c++"
        COMPILE_DEFINITIONS "SWAPTUBE_CPU_DEVICE_TU"
        INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/src/CPUBackend"
    )
endif()

add_compile_options(-fdiagnostics-color=always)

add_executable(${PROJECT_NAME} ${SOURCES} ${ACTIVE_PROJECT_SOURCE} ${GPU_SOURCES})
message(STATUS "Executable target ${PROJECT_NAME} added")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}
    ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS} ${GDKPIXBUF_INCLUDE_DIRS} ${GOBJECT_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIR}
)

# pkg-config reports the library search paths separately from the library names;
# outside /usr/lib the linker needs both.
target_link_directories(${PROJECT_NAME} PRIVATE
    ${FFMPEG_LIBRARY_DIRS} ${RSVG_LIBRARY_DIRS} ${GLIB_LIBRARY_DIRS}
    ${CAIRO_LIBRARY_DIRS} ${GDKPIXBUF_LIBRARY_DIRS} ${GOBJECT_LIBRARY_DIRS}
    ${LZMA_LIBRARY_DIRS}
)

# These pkg-config results overlap heavily (librsvg alone pulls in cairo, glib,
# gobject and lzma), and the duplicates make the linker complain.
set(SWAPTUBE_LINK_LIBRARIES
    ${GLIB_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${RSVG_LIBRARIES}
    ${FDKAAC_LIBRARIES}
    ${CAIRO_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    ${LZMA_LIBRARIES}
    ${PNG_LIBRARIES}
    ${GPU_LIBRARIES}
    ${X11_LIBRARIES}
)
list(REMOVE_DUPLICATES SWAPTUBE_LINK_LIBRARIES)

target_link_libraries(${PROJECT_NAME} ${SWAPTUBE_LINK_LIBRARIES} Threads::Threads)

message(STATUS "Link libraries configured for ${PROJECT_NAME}")
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)
message(STATUS "Configuration complete.")
