cmake_minimum_required(VERSION 3.10)
project(Model3D)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

file(GLOB SRC_FILES src/*.cpp)
include_directories(./include)
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(libigl CONFIG REQUIRED)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL Eigen3::Eigen igl::core igl::common)