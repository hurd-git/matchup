cmake_minimum_required(VERSION 3.25)

project(matchup LANGUAGES C CXX)

if(NOT WIN32)
    message(FATAL_ERROR "matchup currently supports Windows only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(MATCHUP_OPENCV_SOURCE_DIR "" CACHE PATH
    "Optional local OpenCV source tree used instead of downloading OpenCV"
)

# Only build the OpenCV modules used by the matcher. Python bindings, media
# codecs, GUI, DNN, tests and applications are not part of the wheel.
set(BUILD_LIST core,imgproc,features2d,calib3d,flann CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_dnn OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_highgui OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_imgcodecs OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_ml OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_objdetect OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_photo OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_video OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_videoio OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
set(WITH_MSMF OFF CACHE BOOL "" FORCE)
set(WITH_DSHOW OFF CACHE BOOL "" FORCE)
set(WITH_AVIF OFF CACHE BOOL "" FORCE)
set(WITH_JASPER OFF CACHE BOOL "" FORCE)
set(WITH_JPEG OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_OPENJPEG OFF CACHE BOOL "" FORCE)
set(WITH_PNG OFF CACHE BOOL "" FORCE)
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_VTK OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_OPENCLAMDBLAS OFF CACHE BOOL "" FORCE)
set(WITH_OPENCLAMDFFT OFF CACHE BOOL "" FORCE)
set(WITH_IPP ON CACHE BOOL "" FORCE)
set(BUILD_IPP_IW ON CACHE BOOL "" FORCE)
set(WITH_TBB OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_PTHREADS_PF OFF CACHE BOOL "" FORCE)
set(WITH_EIGEN OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(OPENCV_DISABLE_THREAD_SUPPORT ON CACHE BOOL "" FORCE)
set(OPENCV_SKIP_MSVC_PARALLEL ON CACHE BOOL "" FORCE)
set(OPENCV_ENABLE_ALLOCATOR_STATS OFF CACHE BOOL "" FORCE)
set(OPENCV_GENERATE_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(CPU_BASELINE SSE3 CACHE STRING "" FORCE)
set(CPU_DISPATCH "SSE4_1;SSE4_2;AVX;FP16;AVX2" CACHE STRING "" FORCE)

if(MATCHUP_OPENCV_SOURCE_DIR)
    get_filename_component(
        MATCHUP_OPENCV_SOURCE_DIR
        "${MATCHUP_OPENCV_SOURCE_DIR}"
        ABSOLUTE
    )
    if(NOT EXISTS "${MATCHUP_OPENCV_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "MATCHUP_OPENCV_SOURCE_DIR is not an OpenCV source tree: "
            "${MATCHUP_OPENCV_SOURCE_DIR}"
        )
    endif()
    set(MATCHUP_OPENCV_BINARY_DIR "${CMAKE_BINARY_DIR}/opencv-build")
else()
    include(FetchContent)
    FetchContent_Populate(
        opencv
        URL "https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz"
        URL_HASH "SHA256=44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(MATCHUP_OPENCV_SOURCE_DIR "${opencv_SOURCE_DIR}")
    set(MATCHUP_OPENCV_BINARY_DIR "${opencv_BINARY_DIR}")
endif()

add_subdirectory(
    "${MATCHUP_OPENCV_SOURCE_DIR}"
    "${MATCHUP_OPENCV_BINARY_DIR}"
    EXCLUDE_FROM_ALL
)

if(TARGET ipphal)
    target_include_directories(ipphal PRIVATE
        "${MATCHUP_OPENCV_SOURCE_DIR}/include"
        "${MATCHUP_OPENCV_SOURCE_DIR}/modules/core/include"
        "${CMAKE_BINARY_DIR}"
        "${MATCHUP_OPENCV_BINARY_DIR}"
    )
endif()

pybind11_add_module(_native MODULE NO_EXTRAS cpp/matchup_native.cpp)
target_compile_definitions(_native PRIVATE
    PY_SSIZE_T_CLEAN
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)
if(MSVC)
    target_compile_options(_native PRIVATE /O2 /EHsc /W4)
endif()
target_include_directories(_native PRIVATE
    "${MATCHUP_OPENCV_SOURCE_DIR}/include"
    "${MATCHUP_OPENCV_SOURCE_DIR}/modules/core/include"
    "${MATCHUP_OPENCV_SOURCE_DIR}/modules/imgproc/include"
    "${MATCHUP_OPENCV_SOURCE_DIR}/modules/features2d/include"
    "${MATCHUP_OPENCV_SOURCE_DIR}/modules/calib3d/include"
    "${MATCHUP_OPENCV_SOURCE_DIR}/modules/flann/include"
    "${CMAKE_BINARY_DIR}"
    "${MATCHUP_OPENCV_BINARY_DIR}"
)
target_link_libraries(_native PRIVATE
    opencv_calib3d
    opencv_features2d
    opencv_imgproc
    opencv_flann
    opencv_core
)

install(TARGETS _native
    LIBRARY DESTINATION matchup
    RUNTIME DESTINATION matchup
)
foreach(opencv_target IN ITEMS
    opencv_core
    opencv_imgproc
    opencv_features2d
    opencv_flann
    opencv_calib3d
)
    install(FILES "$<TARGET_FILE:${opencv_target}>" DESTINATION matchup)
endforeach()
