cmake_minimum_required(VERSION 3.8)
project(aruco_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# colcon builds with no build type by default, which means no optimisation.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect aruco)
find_package(yaml-cpp REQUIRED)

# yaml-cpp 0.7 exports "yaml-cpp", 0.8 prefers "yaml-cpp::yaml-cpp"
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

# ROS-free detection core (OpenCV only) so it can be reused / unit-tested on its own
add_library(aruco_tracker_core STATIC src/marker_detector.cpp)
target_include_directories(aruco_tracker_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aruco_tracker_core PUBLIC ${OpenCV_LIBS})

add_executable(aruco_detector_node src/aruco_detector_node.cpp src/pose_estimator.cpp)
ament_target_dependencies(aruco_detector_node
  rclcpp sensor_msgs geometry_msgs std_msgs tf2 tf2_ros)
target_link_libraries(aruco_detector_node aruco_tracker_core ${YAML_CPP_TARGET} ${OpenCV_LIBS})

install(TARGETS aruco_detector_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
