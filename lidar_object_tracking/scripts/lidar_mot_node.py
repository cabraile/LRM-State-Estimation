#!/usr/bin/env python3
import os
from typing import Tuple
import numpy as np
import yaml

from scipy.spatial.transform import Rotation

import rospy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros

from modules.interface import tf_transform_to_transformation_matrix, ros_pose_from_transformation_matrix, ros_pose_to_transformation_matrix
from modules.bounding_box import BoundingBox2D
from modules.tracker import Sort

class MOTNode:

    def load_settings(self) -> None:
        project_dir = os.path.join(os.path.dirname(__file__),os.pardir)
        settings_path = os.path.join(project_dir, "config", "mot_tracker.yaml")
        with open(settings_path,"r") as settings_file:
            self.settings_dict = yaml.safe_load(settings_file)

    def __init__(self) -> None:
        self.load_settings()
        sort_settings = self.settings_dict["sort"]
        self.mot = Sort(
            max_age         = sort_settings["max_age"], 
            min_hits        = sort_settings["min_hits"],
            iou_threshold   = sort_settings["iou_threshold"]
        )
        self.utm_T_lidar = None

    def wait_for_extrinsics(self) -> None:
        # tf_buffer = tf2_ros.Buffer()
        # tf_listener = tf2_ros.TransformListener(tf_buffer)
        # while not rospy.is_shutdown():
        #     try:
        #         # http://docs.ros.org/en/indigo/api/tf2_ros/html/c++/classtf2__ros_1_1Buffer.html
        #         # According to the docs it the argument sequence is target,source and time
        #         transform_base_link_T_lidar = tf_buffer.lookup_transform("base_link", "velodyne", rospy.Time(0))
        #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #         continue
        #self.base_link_T_lidar = tf_transform_to_transformation_matrix(transform_base_link_T_lidar)
        self.base_link_T_lidar = np.eye(4)

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
            width = bounding_box.scale.x
            length = bounding_box.scale.y
            height = bounding_box.scale.z
            # TODO Fetch the score
            score = 1.0
            # Create bounding box
            # TODO tracker does not use compute the yaw
            delta_x = width# * np.cos(yaw)
            delta_y = length# * np.sin(yaw)
            bounding_box = BoundingBox2D.from_center_and_size(x_center, y_center, delta_x, delta_y)
            x_left, y_bottom, x_right, y_top = bounding_box.get_corners()
            detection_bounding_boxes.append( np.array([x_left,y_bottom, x_right, y_top, score]) )
        detections_array = np.empty((0,5))
        if len(detection_bounding_boxes) > 0:
            detections_array = np.vstack(detection_bounding_boxes)
        
        # Update the MOT
        tracked_objects_array = self.mot.update(detections_array)
        
        # Convert back the tracked objects to MarkerArray
        tracked_object_markers = []
        for i in range(tracked_objects_array.shape[0]):
            x_left, y_bottom, x_right, y_top, identifier = tracked_objects_array[i]
            identifier = int(identifier)
            bounding_box = BoundingBox2D.from_corners(x_left, y_bottom, x_right, y_top)
            x_center, y_center, width, length = bounding_box.get_as_center_and_size()
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_stamp
            #marker.namespace = "detections"
            marker.id = identifier
            marker.type = Marker.CUBE
            utm_T_object = np.eye(4) # TODO: fill with the orientation
            utm_T_object[0,3] = x_center
            utm_T_object[1,3] = y_center
            marker.pose = ros_pose_from_transformation_matrix(utm_T_object)
            marker.scale.x = width
            marker.scale.y = length
            marker.scale.z = 1.0 # TODO: replace by the dimensions
            r,g,b = self.mot.trackers[i].color
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(2)
            tracked_object_markers.append(marker)

        # Publish output
        tracked_object_markers_msg = MarkerArray()
        tracked_object_markers_msg.markers = tracked_object_markers
        self.pub_tracked_poses.publish(tracked_object_markers)

    def spin(self) -> None:
        self.wait_for_extrinsics()
        
        # Only start communicating when received the first extrinsics message
        node_settings = self.settings_dict["node"]
        self.sub_base_link_pose = rospy.Subscriber(node_settings["in_pose_estimation_topic"], PoseWithCovarianceStamped, self.callback_base_link_pose)
        self.sub_detections = rospy.Subscriber(node_settings["in_detected_objects"], MarkerArray, self.callback_detections)
        self.pub_tracked_poses = rospy.Publisher(node_settings["out_detected_objects"], MarkerArray, queue_size=10)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("lidar_mot_node")
    node = MOTNode()
    node.spin()