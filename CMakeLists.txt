cmake_minimum_required(VERSION 2.8.3)
project(number_detection)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


find_package( OpenCV REQUIRED )


add_library( KNN_number_detection src/KNN_number_detection.cpp )

## Declare a cpp executable
add_executable( testNumDetection src/test/testNumDetcetion.cpp )

## Specify libraries to link a library or executable target against
target_link_libraries( testNumDetection  KNN_number_detection )
target_link_libraries( testNumDetection  ${OpenCV_LIBS} )
