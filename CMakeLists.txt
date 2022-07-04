cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FSS-1000 VERSION 1.0.0 LANGUAGES CXX)
set(EXECUTABLE_NAME FSS-1000)
list(APPEND CMAKE_PREFIX_PATH "/home/vsankepa/projects/dcgan/libtorch_new/libtorch")


find_package(Torch REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/home/vsankepa/projects/dcgan/vision")
find_package(TorchVision  REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/home/vsankepa/projects/dcgan/opencv-4.x")
Set(OpenCV_INCLUDE_DIRS "/home/vsankepa/projects/dcgan/opencv-4.x/include/opencv2/")
message(STATUS "foo include dir: ${OpenCV_INCLUDE_DIRS}")
set(OpenCV_DIR  "/home/vsankepa/projects/dcgan/opencv-4.x/build")
find_package(OpenCV  REQUIRED)
FIND_PACKAGE(Boost)

SET(PYTHON_LIBRARY "/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so")
#find_package(PythonInterp 3.8 REQUIRED)
#find_package(PythonLibs 3.8 REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development)
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/FSS1000_main.cpp
                                          src/CNNEncoder.cpp
                                          includes/CNNEncoder.h

)


target_include_directories(${EXECUTABLE_NAME} PRIVATE includes)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} )

target_link_libraries(${EXECUTABLE_NAME} TorchVision::TorchVision)
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS} )

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)