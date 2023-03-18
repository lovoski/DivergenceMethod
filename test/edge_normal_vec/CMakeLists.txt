cmake_minimum_required(VERSION 3.10)
project(edge_normal_vec)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
file(GLOB SRC_FILES *.cpp)
include_directories(${PROJECT_SOURCE_DIR}/Model3D/include)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen Model3D)