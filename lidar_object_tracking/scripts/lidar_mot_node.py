#!/usr/bin/env python3
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from std_msgs.msg import Bool
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros

from msgs_perception.msg import Obstacle,ObstacleArray
from modules.interface import   tf_transform_to_transformation_matrix, \
                                ros_pose_from_transformation_matrix, \
                                ros_pose_to_transformation_matrix, \
                                invert_transformation_matrix

from modules.tracker import Sort

class MOTNode:

    def load_settings(self) -> None:
        project_dir = os.path.join(os.path.dirname(__file__),os.pardir)
        settings_path = os.path.join(project_dir, "config", "mot_tracker.yaml")
        with open(settings_path,"r") as settings_file:
            self.settings_dict = yaml.safe_load(settings_file)

    def __init__(self) -> None:
        self.load_settings()
        node_settings = self.settings_dict["node"]
        sort_settings = self.settings_dict["sort"]
        #prediction_rate = node_settings["lidar"]["expected_frame_rate"] # TODO: switch between sensors (max between the frame rates)
        full_param_name = rospy.search_param("expected_frame_rate")
        prediction_rate = rospy.get_param(full_param_name)
        self.mot = Sort(
            max_age         = sort_settings["max_age"], 
            min_hits        = sort_settings["min_hits"],
            iou_threshold   = sort_settings["iou_threshold"],
            prediction_rate = prediction_rate
        )
        self.utm_T_lidar = None

    def wait_for_extrinsics(self) -> None:
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        while not rospy.is_shutdown():
            try:
                # http://docs.ros.org/en/indigo/api/tf2_ros/html/c++/classtf2__ros_1_1Buffer.html
                # According to the docs it the argument sequence is target,source and time
                transform_base_link_T_lidar = tf_buffer.lookup_transform("base_link", "velodyne", rospy.Time(0))
                # TODO lookup other sensors if necessary
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        self.base_link_T_lidar = tf_transform_to_transformation_matrix(transform_base_link_T_lidar)

    def callback_base_link_pose(self, msg : PoseWithCovarianceStamped) -> None:
        self.utm_T_base_link = ros_pose_to_transformation_matrix(msg.pose.pose)
        self.utm_T_lidar = self.utm_T_base_link @ self.base_link_T_lidar

    def callback_detections(self, msg : MarkerArray) -> None:
        if self.utm_T_lidar is None:
            return
        current_stamp = None # In theory, should use the most recent stamp, but will keep like this for simplifying for now
        # Load all the bounding boxes from the marker array
        detection_bounding_boxes = []
        for bounding_box in msg.markers:
            current_stamp = bounding_box.header.stamp
            lidar_T_object = ros_pose_to_transformation_matrix(bounding_box.pose)

            # Coordinates
            utm_T_object = self.utm_T_lidar @ lidar_T_object
            x_center,y_center,z_center = utm_T_object[:3,3]

            # Orientation
            object_rotation = Rotation.from_matrix(utm_T_object[:3,:3])
            roll, pitch, yaw = object_rotation.as_euler("xyz",degrees=False)

            # Size
            delta_x = bounding_box.scale.x
            delta_y = bounding_box.scale.y
            delta_z = bounding_box.scale.z

            # TODO Fetch the score
            score = 1.0

            # Create bounding box
            detection_bounding_boxes.append( np.array([x_center,y_center, z_center, delta_x, delta_y, delta_z, yaw, score]) )

        detections_array = np.empty((0,8))
        if len(detection_bounding_boxes) > 0:
            detections_array = np.vstack(detection_bounding_boxes)

        if detection_bounding_boxes == []: #error when list es empty
            # Publish output
            tracked_object_markers_msg = MarkerArray()
            tracked_object_markers_msg.markers = []
            self.pub_lidar_tracked_markers.publish(tracked_object_markers)

            tracked_object_obstacles_msg = ObstacleArray()
            # tracked_object_obstacles_msg.header = tracked_object_obstacles[-1].header       #error when list is empty
            tracked_object_obstacles_msg.obstacle = []
            self.pub_lidar_tracked_obstacles.publish(tracked_object_obstacles_msg)# always publishing 
            return
        
        # Update the MOT
        tracked_objects_array = self.mot.update(detections_array)
        
        # Convert back the tracked objects to MarkerArray
        tracked_object_markers = []
        tracked_object_obstacles = []
        
        for i in range(tracked_objects_array.shape[0]):
            marker_msg = Marker()

            # Tracked object to Marker
            utm_T_object = np.eye(4)
            x_center, y_center, z_center, delta_x, delta_y, delta_z, yaw, vx_utm, vy_utm, vz_utm, identifier = tracked_objects_array[i]
            utm_T_object[:3,3] = [x_center,y_center,z_center]
            utm_T_object[:3,:3] = Rotation.from_euler("z", [yaw], degrees=False).as_matrix()
            marker_msg.pose = ros_pose_from_transformation_matrix(utm_T_object)
            marker_msg.scale.x = delta_x
            marker_msg.scale.y = delta_y
            marker_msg.scale.z = delta_z

            # Marker metadata
            marker_msg.header.frame_id = "map"
            marker_msg.header.stamp = current_stamp
            marker_msg.type = Marker.CUBE
            identifier = int(identifier)
            marker_msg.id = identifier
            r,g,b = self.mot.get_tracker_by_id(identifier).color
            marker_msg.color.r = r
            marker_msg.color.g = g
            marker_msg.color.b = b
            marker_msg.color.a = 0.6
            marker_msg.lifetime = rospy.Duration(0.35)

            # Store Marker
            tracked_object_markers.append(marker_msg)

            # Tracked object to Obstacle message
            obstacle_msg = Obstacle()
            obstacle_msg.header.frame_id = "velodyne"
            obstacle_msg.header.stamp = marker_msg.header.stamp
            lidar_T_utm = invert_transformation_matrix(self.utm_T_lidar)
            lidar_T_object = lidar_T_utm @ utm_T_object
            obstacle_msg.pose = ros_pose_from_transformation_matrix(lidar_T_object)
            object_linear_velocity_wrt_utm = np.array([vx_utm, vy_utm, vz_utm])
            object_linear_velocity_wrt_lidar = (lidar_T_utm[:3,:3] @ object_linear_velocity_wrt_utm.reshape(-1,1)).flatten()
            obstacle_msg.twist.linear.x = object_linear_velocity_wrt_lidar[0]
            obstacle_msg.twist.linear.y = object_linear_velocity_wrt_lidar[1]
            obstacle_msg.twist.linear.z = object_linear_velocity_wrt_lidar[2]
            obstacle_msg.scale = marker_msg.scale
            obstacle_msg.color = marker_msg.color
            obstacle_msg.lifetime = marker_msg.lifetime
            obstacle_msg.frame_locked = marker_msg.frame_locked
            obstacle_msg.action = marker_msg.action
            obstacle_msg.type = marker_msg.type
            obstacle_msg.id = identifier

            # Store obstacle
            tracked_object_obstacles.append(obstacle_msg)

        # Publish output
        tracked_object_markers_msg = MarkerArray()
        tracked_object_markers_msg.markers = tracked_object_markers
        self.pub_lidar_tracked_markers.publish(tracked_object_markers)

        tracked_object_obstacles_msg = ObstacleArray()
        # tracked_object_obstacles_msg.header = tracked_object_obstacles[-1].header       #error when list is empty
        tracked_object_obstacles_msg.obstacle = tracked_object_obstacles
        self.pub_lidar_tracked_obstacles.publish(tracked_object_obstacles_msg)

    def shutdown_cb(self, msg):
        if msg.data:
            print ("Bye!")

            self.mot=None
            self.base_link_T_lidar = None
            self.utm_T_base_link = None
            self.utm_T_lidar = None
            self.settings_dict= None

            del self.sub_base_link_pose
            del self.sub_lidar_detections 
            del self.pub_lidar_tracked_markers 
            del self.pub_lidar_tracked_obstacles 
            del self.shutdown_sub
            rospy.signal_shutdown("lidar mot node finished ...")


    def spin(self) -> None:
        self.wait_for_extrinsics()
        
        # Only start communicating when received the first extrinsics message
        node_settings = self.settings_dict["node"]
        lidar_node_settings         = node_settings["lidar"]
        self.sub_base_link_pose     = rospy.Subscriber(node_settings["in_pose_estimation_topic"], PoseWithCovarianceStamped, self.callback_base_link_pose)


        in_detected_objects_full_name = rospy.search_param("in_detected_objects")
        in_detected_objects = rospy.get_param(in_detected_objects_full_name)
        out_detected_objects_markers_full_name = rospy.search_param("out_detected_objects_markers")
        out_detected_objects_markers = rospy.get_param(out_detected_objects_markers_full_name)
        out_detected_objects_obstacles_full_name = rospy.search_param("out_detected_objects_obstacles")
        out_detected_objects_obstacles = rospy.get_param(out_detected_objects_obstacles_full_name)

        print(in_detected_objects, out_detected_objects_markers, out_detected_objects_obstacles)
        # self.sub_lidar_detections   = rospy.Subscriber(lidar_node_settings["in_detected_objects"], MarkerArray, self.callback_detections)
        # self.pub_lidar_tracked_markers    = rospy.Publisher(lidar_node_settings["out_detected_objects_markers"], MarkerArray, queue_size=10)
        # self.pub_lidar_tracked_obstacles  = rospy.Publisher(lidar_node_settings["out_detected_objects_obstacles"], ObstacleArray, queue_size=10)
        self.sub_lidar_detections   = rospy.Subscriber(in_detected_objects, MarkerArray, self.callback_detections)
        self.pub_lidar_tracked_markers    = rospy.Publisher(out_detected_objects_markers, MarkerArray, queue_size=10)
        self.pub_lidar_tracked_obstacles  = rospy.Publisher(out_detected_objects_obstacles, ObstacleArray, queue_size=10)
                

        self.shutdown_sub = rospy.Subscriber('/carina/vehicle/shutdown', Bool, self.shutdown_cb, queue_size=1)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("mot_node")
    node = MOTNode()
    node.spin()