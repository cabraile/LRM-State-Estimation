import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, TransformStamped

def invert_transformation_matrix( T : np.ndarray ) -> np.ndarray:
    R = T[:3,:3]
    t = T[:3,3]
    T_inv = np.eye(4)
    T_inv[:3,:3] = R.T
    T_inv[:3,3] = - R.T @ t
    return T_inv

def build_transformation_matrix( translation : Tuple[float], orientation_quaternion : Tuple[float] ) -> np.ndarray:
    """
    Arguments
    ---------
    translation: tuple.
        The x,y,z coordinates' tuple.
    orientation_quaternion: tuple.
        The qx,qy,qz,qw tuple.
    
    Returns
    ---------
    The transformation matrix.
    """
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat(orientation_quaternion).as_matrix()
    T[:3, 3] = translation
    return T

def ros_pose_from_transformation_matrix( T : np.ndarray ) -> Pose:
    ros_pose = Pose()
    ros_pose.position.x = T[0,3]
    ros_pose.position.y = T[1,3]
    ros_pose.position.z = T[2,3]
    qx,qy,qz,qw = Rotation.from_matrix(T[:3,:3]).as_quat()
    ros_pose.orientation.x = qx
    ros_pose.orientation.y = qy
    ros_pose.orientation.z = qz
    ros_pose.orientation.w = qw
    return ros_pose

def ros_pose_to_transformation_matrix( ros_pose : Pose ) -> np.ndarray:
    x = ros_pose.position.x
    y = ros_pose.position.y
    z = ros_pose.position.z
    qx = ros_pose.orientation.x
    qy = ros_pose.orientation.y
    qz = ros_pose.orientation.z
    qw = ros_pose.orientation.w
    return build_transformation_matrix(translation=(x,y,z),orientation_quaternion=(qx,qy,qz,qw))

def tf_transform_to_transformation_matrix( tf_transform : TransformStamped ) -> np.ndarray:
    x = tf_transform.transform.translation.x
    y = tf_transform.transform.translation.y
    z = tf_transform.transform.translation.z
    qx = tf_transform.transform.rotation.x
    qy = tf_transform.transform.rotation.y
    qz = tf_transform.transform.rotation.z
    qw = tf_transform.transform.rotation.w
    return build_transformation_matrix(translation=(x,y,z),orientation_quaternion=(qx,qy,qz,qw))
