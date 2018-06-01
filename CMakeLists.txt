project(AffineTransformationEstimation)
cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_FLAGS "-lboost_system -lopencv_nonfree -lpthread -fpermissive")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC ON)


FIND_PACKAGE(VISP REQUIRED)
IF(VISP_FOUND)
INCLUDE(${VISP_USE_FILE})
ENDIF(VISP_FOUND)


SET(SOURCES
main.cpp
 )


add_executable (
  AffineTransformationEstimation ${SOURCES}
)

target_link_libraries(AffineTransformationEstimation ${OpenCV_LIBRARIES}  opencv_nonfree
)






