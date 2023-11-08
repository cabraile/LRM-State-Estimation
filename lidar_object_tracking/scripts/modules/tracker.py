"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from typing import Tuple

import numpy as np

from filterpy.kalman import KalmanFilter

from modules.bounding_box import BoundingBox3D

def linear_assignment(cost_matrix : np.ndarray) -> np.ndarray:
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test : np.ndarray, bb_gt : np.ndarray) -> np.ndarray:
  """
  From SORT: Computes IOU between two batches of bboxes in the form [x1,y1,x2,y2]

  Arguments
  ----------
  bb_test: numpy.ndarray.
    The 2D-array of the detected bounding boxes, in which each row cooresponds to a (x1,y1,x2,y2) bounding box.
  bb_gt: numpy.ndarray.
    The 2D-array of the tracked bounding boxes, in which each row cooresponds to a (x1,y1,x2,y2) bounding box.
  
  Returns
  -----------
  float.
    The IOU score matrixs for all the bounding boxes' pairs across batches.
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox : np.ndarray, prediction_rate : float = 1.0):
    """
    Initialises a tracker using initial bounding box.

    The state vector is in the form: xc,yc,zc,size_x,size_y_size_z,yaw,vx,vy,vz.

    Arguments
    ----------
    bbox : numpy.ndarray.
      The state vector of size 7 containing [xc,yc,zc,size_x,size_y_size_z,yaw].
    prediction_rate: float.
      The frequency in which the predictions are performed (in Hertz).
    """
    #define constant velocity model
    delta_T = 1./prediction_rate
    self.kf = KalmanFilter(dim_x=10, dim_z=7) 
    self.kf.F = np.array([
      [1,0,0,0,0,0,0,delta_T,0,0], # xc
      [0,1,0,0,0,0,0,0,delta_T,0], # yc
      [0,0,1,0,0,0,0,0,0,delta_T], # zc
      [0,0,0,1,0,0,0,0,0,0], # size_x
      [0,0,0,0,1,0,0,0,0,0], # size_y
      [0,0,0,0,0,1,0,0,0,0], # size_z
      [0,0,0,0,0,0,1,0,0,0], # yaw
      [0,0,0,0,0,0,0,1,0,0], # vx
      [0,0,0,0,0,0,0,0,1,0], # vy
      [0,0,0,0,0,0,0,0,0,1], # vz
    ])
    self.kf.H = np.array([
      [1,0,0,0,0,0,0,0,0,0], # xc
      [0,1,0,0,0,0,0,0,0,0], # yc
      [0,0,1,0,0,0,0,0,0,0], # zc
      [0,0,0,1,0,0,0,0,0,0], # size_x
      [0,0,0,0,1,0,0,0,0,0], # size_y
      [0,0,0,0,0,1,0,0,0,0], # size_z
      [0,0,0,0,0,0,1,0,0,0], # yaw
    ])

    self.kf.P *= 10.
    self.kf.P[7:,7:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.R[7:,7:] *= 1. # Measurement uncertainty
    self.kf.R[6,6] = 0.00001 # Set a very low value to the yaw uncertainty in order to avoid 'spinning' behaviour
    self.kf.Q[7:,7:] *= 0.3 # Prediction uncertainty

    self.kf.x[:7] = bbox.reshape(-1,1)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.color = list(np.random.rand(3))
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox : np.ndarray) -> None:
    """
    Updates the state vector with observed bbox.

    Arguments
    ----------
    bbox : numpy.ndarray.
      The state vector of size 8 containing [xc,yc,zc,size_x,size_y_size_z,yaw,score].
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(bbox.reshape(-1,1))

  def predict(self) -> np.ndarray:
    """
    Initialises a tracker using initial bounding box.

    The state vector is in the form: xc,yc,zc,size_x,size_y_size_z,yaw,vx,vy,vz.

    Arguments
    ----------
    bbox : numpy.ndarray.
      The state vector of size 7 containing [xc,yc,zc,size_x,size_y_size_z,yaw].
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x.squeeze()

def associate_detections_to_trackers(detections : np.ndarray, trackers : np.ndarray ,iou_threshold : float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,7),dtype=int)

  # Convert detections to 2D bounding boxes
  detections_2D_format = []
  for i in range(detections.shape[0]):
    xc, yc, zc, size_x, size_y, size_z = detections[i,:6]
    detection_2D_corners = np.array(BoundingBox3D.from_center_and_size(xc,yc,zc, size_x, size_y, size_z).to_bounding_box_2D().get_corners())
    detections_2D_format.append(detection_2D_corners)

  if detections_2D_format == []:# following lines need at least one array to concatenate
    return
    
  detections_2D_format = np.vstack(detections_2D_format)
    
  # Convert trackers to 2D bounding boxes
  trackers_2D_format = []
  for i in range(trackers.shape[0]):
    xc, yc, zc, size_x, size_y, size_z = trackers[i,:6]
    tracker_2D_corners = np.array(BoundingBox3D.from_center_and_size(xc,yc,zc, size_x, size_y, size_z).to_bounding_box_2D().get_corners())
    trackers_2D_format.append(tracker_2D_corners)
  trackers_2D_format = np.vstack(trackers_2D_format) 

  # Computes the 2D IOU between the detected and tracked bounding boxes
  iou_matrix = iou_batch(detections_2D_format, trackers_2D_format)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age : int =1, min_hits : int =3, iou_threshold : float =0.3 , prediction_rate : float = 1.0):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.tracker_id_to_seq_idx_map = dict()
    self.frame_count = 0
    self.prediction_rate = prediction_rate

  def refresh_indices(self) -> None:
    """Refreshes the indices of the tracker ID to its index among the SORT trackers.
    """
    self.tracker_id_to_seq_idx_map = { tracker.id + 1 : idx for idx, tracker in enumerate(self.trackers) }

  def get_tracker_by_id(self, tracker_id : int) -> KalmanBoxTracker:
    return self.trackers[ self.tracker_id_to_seq_idx_map[tracker_id] ]

  def update(self, in_detections : np.ndarray) -> np.ndarray:
    """
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 7)) for frames without detections).

    Arguments
    ----------
    dets: numpy.ndarray.
        A numpy array of detections in the format 
        ```
        [
            [xc,yc,zc,size_x,size_y,size_z,yaw,score],
            [xc,yc,zc,size_x,size_y,size_z,yaw,score],
        ...].
        ```
    Returns
    ----------
    A array similar to the input, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    dets = in_detections[:,:-1] # TODO: reinclude score
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 7))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()
      trk[:] = pos.squeeze()[:7]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], self.prediction_rate)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    
    self.refresh_indices()
    
    if(len(ret)>0):
      return np.concatenate(ret)
    
    return np.empty((0,8))

