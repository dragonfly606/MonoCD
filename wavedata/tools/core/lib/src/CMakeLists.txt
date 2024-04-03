cmake_minimum_required (VERSION 2.6)
project (object_proposals)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_library(integral_images_3d SHARED src/integral_images_3d.cpp)
