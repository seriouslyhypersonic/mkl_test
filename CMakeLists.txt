cmake_minimum_required(VERSION 3.13)
project(mkl_test)

# ------------------------------------------------------------------------------
# Intel MKL
# ------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)

find_package(MKL)
if (MKL_FOUND)
    message(STATUS "Search summary")
    message(STATUS "    MKL interface library        ${MKL_INTERFACE_LIBRARY}")
    message(STATUS "    MKL core library             ${MKL_CORE_LIBRARY}")
    message(STATUS "    MKL threading layer library  ${MKL_THREADING_LAYER_LIBRARY}")
    message(STATUS "    MKL include                  ${MKL_INCLUDE_DIRS}")
    message(STATUS "    MKL libraries                ${MKL_LIBRARIES}")
endif ()

add_executable(mkl_test main.cpp)

target_include_directories(mkl_test PRIVATE ${MKL_INCLUDE_DIRS})

target_link_libraries(mkl_test ${MKL_LIBRARIES})
