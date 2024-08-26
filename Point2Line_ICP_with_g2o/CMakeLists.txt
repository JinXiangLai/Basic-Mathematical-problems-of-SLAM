cmake_minimum_required(VERSION 3.10)
# ROS 的g2o会污染现版本，需要把 source ros关闭
project(p2l_icp)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic) # ljx
endif ()

IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

add_compile_options(-Werror=return-type)

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

add_executable(p2l_icp point_to_line_icp.cpp)

file(GLOB g2oLIBRARIES "/home/laijinxiang/g2o-library/lib/*")
# file(GLOB G2O_LIBRARIES "/home/laijinxiang/ORB_SLAM3_ros2/Thirdparty/g2o/lib/*")

# fmt 是 Sophus依赖库格式化输出使用
target_link_libraries(${PROJECT_NAME} ${g2oLIBRARIES} fmt)
