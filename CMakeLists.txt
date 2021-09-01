cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dl_inference)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS -std=c++17)

include(CTest)
enable_testing()
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(ExternalProject)
include(cmake/Utils.cmake)

option(BUILD_ZLIB  "Build zlib from source"  OFF )
option(BUILD_JSON  "Build json from source"  ON )
option(BUILD_BOOST  "Build boost from source"  OFF )
option(BUILD_TEST "Build test" OFF )
option(BUILD_SAMPLES "Build samples" ON )
option(INSTALL_DATA "Build samples" ON )

#try_compile(HAS_FS "${CMAKE_BINARY_DIR}/temp"
#        "${CMAKE_SOURCE_DIR}/test/sample.cpp" CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
#        LINK_LIBRARIES stdc++fs)
#if(HAS_FS)
#    message(-- STATUS "Compiler has filesystem support")
#else()
#    #   .... You could also try searching for boost::filesystem here.
#    message("-- Compiler is missing filesystem capabilities, set file system on")
#    option(BUILD_BOOST  "Build json from source"  ON )
#endif(HAS_FS)

# 几点注意的:
#1.inline function 如果修改了记得需要先clean 掉环境然后在重新build.
#2.可以通过add_definition的方式增加 #define中定义的变量，如下面的DEBUG
#3.在修改目录后由于是与远程的关系，有些与远程h文件如果不对应的清理的话，很可能找到错误的头文件，使得build时报错un defined reference
#4.注意一定在头文件的定义中加入#ifndef __TR_CONFIG_H__类似的语句，否则会导致头文件import冲突的问题.


if (BUILDE_ZLIB)
    include(cmake/zlib.cmake)
endif()

if (BUILD_JSON)
    include(cmake/json.cmake)
endif ()

if (BUILD_BOOST)
    include(cmake/boost.cmake)
endif ()


#set(CMAKE_PREFIX_PATH  /data/duser/cpp_libs/libtorch)
set(CMAKE_PREFIX_PATH  /usr/local/cpp_libs/libtorch)
#this is for chinese vocab
set(ENV{LC_ALL} "zh_CN.UTF-8")
find_package(Torch REQUIRED)

include_directories("/usr/local/cpp_libs/opencv/include/opencv4")
#include_directories("/data/duser/cpp_libs/opencv/include/opencv4")

LINK_DIRECTORIES(/usr/local/cpp_libs/opencv/lib)
#LINK_DIRECTORIES(/data/duser/cpp_libs/opencv/lib)

set(OPENCV_LIBRARIES opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)

#add_definitions(-DDEBUG) # use this to add define DEBUG
#add_definitions(-DDEBUG_TIME)

add_subdirectory(modules)

add_subdirectory(apps)

if (BUILD_TEST)
    add_subdirectory(test)
endif()

if (BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()

if (INSTALL_DATA)
    install(DIRECTORY data DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT libs)
endif ()










