cmake_minimum_required(VERSION 3.10.0)
project(lrm_lidar_object_tracking VERSION 0.0.1)

find_package(catkin REQUIRED COMPONENTS 
    rospy 
)

catkin_package(INCLUDE_DIRS CATKIN_DEPENDS ${REQUIRED_PACKAGES})