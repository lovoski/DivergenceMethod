cmake_minimum_required(VERSION 3.10)
project(gradient_viewer)
set(CMAKE_CXX_STANDARD 14)

find_package(libigl CONFIG REQUIRED)
include_directories(${libigl_INCLUDE_DIR})
file(GLOB SRC_FILES *.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE igl::core igl::common igl::opengl igl::opengl_glfw)