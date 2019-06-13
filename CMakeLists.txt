project(PRO)
cmake_minimum_required(VERSION 2.8)

INCLUDE_DIRECTORIES(
        include/cnn
        /usr/local/cuda/include
)

add_subdirectory(cuda/cnn ./cuda/cnn/bin)
add_subdirectory(src/cnn ./src/cnn/bin)
set(EXTRA_LIBS ${EXTRA_LIBS} libcudacnn libcnn)
SET(CMAKE_C_COMPILER g++)
list(APPEND CMAKE_C_FLAGS "-lstdc++")

add_executable(pro main.c)
target_link_libraries(pro ${EXTRA_LIBS})