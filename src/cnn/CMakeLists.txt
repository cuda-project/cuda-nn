project(PRINT)
cmake_minimum_required(VERSION 2.8)

#file(GLOB_RECURSE CURRENT_HEADERS *.h)
file(GLOB CURRENT_SOURCES *.c)

set(CMAKE_C_COMPILER g++)
list(APPEND CMAKE_C_FLAGS "-lstdc++")


add_library(libcnn STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})


