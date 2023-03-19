cmake_minimum_required(VERSION 3.10)
project(DivergenceMethod)
set(CMAKE_CXX_STANDARD 14)

find_package(libigl CONFIG REQUIRED)
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)
include_directories(${PROJECT_SOURCE_DIR} Model3D/include)
add_subdirectory(Model3D)
link_directories(Model3D)
add_subdirectory(test/edge_normal_vec)
add_subdirectory(test/gradient_viewer)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE igl::core igl::common igl::opengl igl::opengl_glfw Model3D)