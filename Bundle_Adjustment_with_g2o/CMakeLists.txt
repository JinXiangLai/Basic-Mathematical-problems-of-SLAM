cmake_minimum_required(VERSION 2.8)
# ROS 的g2o会污染现版本，需要把 source ros关闭
project(ba)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
  # SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

IF( CMAKE_BUILD_TYPE==Release )
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
ELSE()
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O ")
ENDIF()
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# set(g2o_DIR "/home/laijinxiang/g2o-library/cmake_modules")
file(GLOB g2oInclude "/home/laijinxiang/g2o-library/g2o/*")
find_package(Eigen3 REQUIRED)
# find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    # ${g2o_INCLUDE_DIR}
    ${g2oInclude}
    ${SOPHUS_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#add_library(utils utils.cpp)
#target_link_libraries(utils ${PCL_LIBRARIES})

add_executable(ba ba.cpp)

file(GLOB g2oLIBRARIES "/home/laijinxiang/g2o-library/lib/*")
# file(GLOB G2O_LIBRARIES "/home/laijinxiang/ORB_SLAM3_ros2/Thirdparty/g2o/lib/*")

# fmt 是 Sophus依赖库格式化输出使用
target_link_libraries(${PROJECT_NAME} ${g2oLIBRARIES} fmt)
