cmake_minimum_required(VERSION 3.8)
project(aruco_plane_localizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(plane_transform_node src/plane_transform_node.cpp)
ament_target_dependencies(plane_transform_node rclcpp geometry_msgs std_msgs)
target_link_libraries(plane_transform_node Eigen3::Eigen yaml-cpp)

install(TARGETS plane_transform_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
