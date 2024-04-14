import tensorflow as tf
import math
from tensorflow.python.ops.numpy_ops import np_config

def iou(bb_a, bb_b):
  xx1 = max(bb_a[0], bb_b[0])
  yy1 = max(bb_a[1], bb_b[1])
  xx2 = min(bb_a[2], bb_b[2])
  yy2 = min(bb_a[3], bb_b[3])
  w = max(0., xx2 - xx1)
  h = max(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_a[2] - bb_a[0])*(bb_a[3] - bb_a[1]) + (bb_b[2] - bb_b[0])*(bb_b[3] - bb_b[1]) - wh)
  return(o)

def distance_loss(bb_a, bb_b):
  """
  Defined as the squared Euclidean distance of the two central points
  divided by square of the diagnal length of the smallest enclosing box
  covering the two boxes.
  """
  w_a = bb_a[2] - bb_a[0]
  w_b = bb_b[2] - bb_b[0]
  h_a = bb_a[3] - bb_a[1]
  h_b = bb_b[3] - bb_b[1]
  x_a = bb_a[2] + w_a/2.
  x_b = bb_b[2] + w_b/2.
  y_a = bb_a[3] + h_a/2.
  y_b = bb_b[3] + h_b/2.
  diag_a = tf.sqrt(tf.pow(w_a, 2)+tf.pow(h_a, 2))
  diag_b = tf.sqrt(tf.pow(w_b, 2)+tf.pow(h_b, 2))
  distance_sq = tf.pow(x_a-x_b, 2)+tf.pow(y_a-y_b, 2)
  loss = distance_sq / tf.pow(tf.sqrt(distance_sq) + diag_a/2 + diag_b/2, 2)
  return loss

def distance_loss_x(x_a, x_b):
  """
  Defined as the squared Euclidean distance of the two central points
  divided by square of the diagnal length of the smallest enclosing box
  covering the two boxes.
  """
  w_a = x_a[2] * x_a[3]
  w_b = x_b[2] * x_b[3]
  h_a = x_a[2] / w_a
  h_b = x_b[2] / w_b
  diag_a = tf.sqrt(tf.pow(w_a, 2)+tf.pow(h_a, 2))
  diag_b = tf.sqrt(tf.pow(w_b, 2)+tf.pow(h_b, 2))
  distance_sq = tf.pow(x_a[0]-x_b[0], 2)+tf.pow(x_a[1]-x_b[1], 2)
  loss = distance_sq / tf.pow(tf.sqrt(distance_sq) + diag_a/2 + diag_b/2, 2)
  return loss

def aspect_ratio_loss(bb_a, bb_b):
  """See page 4 of https://arxiv.org/pdf/1911.08287.pdf"""
  w_a = bb_a[2] - bb_a[0]
  w_b = bb_b[2] - bb_b[0]
  h_a = bb_a[3] - bb_a[1]
  h_b = bb_b[3] - bb_b[1]
  v = 4/tf.pow(math.pi, 2) * tf.pow(tf.atan2(w_a, h_a)-tf.atan2(w_b, h_b), 2)
  return v

def aspect_ratio_loss_x(x_a, x_b):
  w_a = x_a[2] * x_a[3]
  w_b = x_b[2] * x_b[3]
  h_a = x_a[2] / w_a
  h_b = x_b[2] / w_b
  return 4/tf.pow(math.pi, 2) * tf.pow(tf.atan2(w_a, h_a)-tf.atan2(w_b, h_b), 2)