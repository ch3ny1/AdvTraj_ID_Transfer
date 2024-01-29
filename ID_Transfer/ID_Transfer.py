import numpy as np
import tensorflow as tf
import cv2
import math
from ID_Transfer.sort_TF import *
from ID_Transfer.loss import *
from tensorflow.python.ops.numpy_ops import np_config

def update_predict(tracker, z, s=2400, r=1.5):
    """
    Return the KF prior prediction if we update the current tracker with z
    Input: 
        - tracker: KalmanFilterTracker object
        - z: the state vector tf.Tensor([x, y, s, r]), shape=(4, 1)
                or tf.Tensor([x,y]), shape=(2,1)
        - s: scale of the bounding box (w * h)
        - r: aspect ratio of the bounding box
    Output:
        Prior prediction for the next time step
    """
    if z.shape[0] == 2:
        z = tf.concat([z, [[s],[r]]], axis=0)
    elif z.shape[0] == 3:
        z = tf.concat([z, [r]], axis=0)
    elif z.shape[0] == 4:
        z = tf.reshape(z, [4, 1])
    # Update step
    dim_x = tracker.kf.dim_x
    x = tracker.kf.x
    y = tracker.kf.y
    R = tracker.kf.R
    H = tracker.kf.H
    P = tracker.kf.P
    y = z - tf.matmul(H, x)
    PHT = tf.matmul(P, tf.transpose(H))
    S = tf.matmul(H, PHT) + R
    SI = tf.linalg.inv(S)
    K = tf.matmul(PHT, SI)
    x = x + tf.matmul(K, y)
    I_KH = tf.eye(dim_x) - tf.matmul(K, H)
    P = tf.matmul(tf.matmul(I_KH, P), tf.transpose(I_KH)) + tf.matmul(tf.matmul(K, R), tf.transpose(K))
    # Prediction Step
    F = tracker.kf.F
    x = tf.matmul(F, x)
    return x


def convert_to_bbox(x):
    s, r = x[2], x[3]
    w = tf.sqrt(s * r)
    h = s / w
    return tf.stack([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.])

def optimize_bbox(tracker_A, tracker_B, z_A, bbox_B, lr=0.1, iteration=30, threshold=0.35, iou_scale=10):
    """
    Perform gradient descent on the adversarial loss to find next state for the attacker
    Input:
        - tracker_A: KalmanFilterTracker object for the attacker
        - tracker_B: KalmanFilterTracker object for the victim
        - z_A: attacker's previous position (starting point) tf.Tensor([x, y, s, r]), shape=(4,)
        - bbox_B: victim's current KF predicted states in the form of bbox
        - lr: learning rate alpha
        - iteration: max number of iterations
        - threshold: limit the relative intra-frame IoU for the attacker, a rudimentary way to ensure physical realizability
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    z_init = tf.identity(z_A) # To limit physical maneuverbility, prevents teleporting
    bbox_B1 = update_predict(tracker_B, convert_bbox_to_z(bbox_B)) # Victim's KF prediction for the next time step
    ratio = bbox_B1[3] # Fix the aspect ratio, same assumption as SORT
    for it in range(iteration):
        if iou(convert_to_bbox(tf.concat([z_A, [ratio]], axis=0)), convert_to_bbox(tf.concat([z_init, [ratio]], axis=0))) < threshold:
            break
        with tf.GradientTape() as tape:
            z_predicted = update_predict(tracker_A, z_A[:3], r=ratio)[:4]
            #z_predicted = tracker_A.update_predict_no_trace(convert_to_bbox(tf.concat([z_A, [ratio]], axis=0)))
            z_A_temp = tf.concat([z_A, [ratio]], axis=0)
            w = tf.sqrt(z_predicted[2] * z_predicted[3])
            h = z_predicted[2] / w
            iou_loss_1 = iou([z_predicted[0]-w/2.,z_predicted[1]-h/2.,z_predicted[0]+w/2.,z_predicted[1]+h/2.], bbox_B1)
            iou_loss_2 = iou([z_A_temp[0]-w/2.,z_A_temp[1]-h/2.,z_A_temp[0]+w/2.,z_A_temp[1]+h/2.], bbox_B)
            loss = -iou_scale*(iou_loss_1 + iou_loss_2) + (distance_loss_x(z_predicted, convert_bbox_to_z(bbox_B1)) + distance_loss_x(z_A_temp, convert_bbox_to_z(bbox_B))) #+ alpha * v
        grads = tape.gradient(loss, [z_A])
        optimizer.apply_gradients(zip(grads, [z_A]))
    return z_A

def convert_x_to_bbox_batch(xs, r):
    bbox_out = []
    for i, x in enumerate(xs):
        bbox_out.append(convert_to_bbox(tf.concat([x, [r[i]]], axis=0)))
    return np.squeeze(np.array(bbox_out))

def move_covertly(origin, direction, min_iou=0.9, speed=1, s=2400, r=1.5):
    bbox = convert_to_bbox(tf.concat([origin, [[s], [r]]], axis=0))
    W = bbox[2] - bbox[0]
    H = bbox[3] - bbox[1]
    u,v = tf.abs(direction)
    if v == 0:
        origin[0].assign(tf.cast(origin[0] + (direction[0]/u)*(1-min_iou)*W*H/((1+min_iou)*H), dtype=tf.float32))
    elif u == 0:
        origin[1].assign(tf.cast(origin[1] + (direction[1]/v)*(1-min_iou)*W*H/((1+min_iou)*W), dtype=tf.float32))
    else:
        a = (min_iou + 1) * u * v
        b = (min_iou + 1)*(u*H+v*W)
        c = (1 - min_iou)*W*H
        delta = tf.pow(b,2) - 4*a*c
        if delta < 0:
            return origin
        else:
            d = (-b + tf.sqrt(delta)) / (2*a)
            origin[0].assign(tf.cast(origin[0] - d*direction[0], dtype=tf.float32))
            origin[1].assign(tf.cast(origin[1] - d*direction[1], dtype=tf.float32))

def trajGen2(adv_init, vctm_traj, lr=1, iter=50, escape_dir=(0,0), threshold=0.05, patience=0, s=2400, r=1.5):
    """
    Input:
        adv_init: initial attacker bbox position (x1,y1,x2,y2)
        vctm_traj: array of victim bboxes representing the trajectory
        escape_dir: desired escape direction for the attcker, after ID-switch
        threshold: min IoU we require the attacker to have intra-frame
        patience: number of frames after initial ID-switch to wait before escape
        s, r: scale and aspect ratio of the attacker
    """
    length = vctm_traj.shape[0]
    adv_z = convert_bbox_to_z(adv_init)
    aspect_ratio = [adv_z[3]]
    z0 = tf.Variable(adv_z[:3])
    traj_gen = [tf.identity(z0)]
    adv_tracker = KalmanBoxTracker(adv_init)
    vctm_tracker = KalmanBoxTracker(vctm_traj[0])
    count = 0
    stop_pos = None
    switched = -1
    for i in range(1, length):
        vctm_curr = vctm_traj[i]
        r = convert_bbox_to_z(vctm_curr)[3]
        aspect_ratio.append(r)
        vctm_pred = vctm_tracker.predict()[0]
        adv_pred = adv_tracker.predict_no_trace()[0]
        adv_curr = convert_x_to_bbox(tf.concat([traj_gen[-1], [r]], axis=0))[0]
        ious = iou_batch([vctm_curr, adv_curr], [vctm_pred, adv_pred])
        print(ious)
        if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] and count >= patience: #ID already switched
            if switched == -1:
                switched = i
            if escape_dir == (0,0):
                if stop_pos is None:
                    z0 = tf.Variable(convert_bbox_to_z(vctm_pred)[:3])
                    stop_pos = tf.Variable(tf.identity(z0))
                else:
                    z0 = stop_pos
            else:
                iou_diff = max(ious[0,0], 0.95)
                move_covertly(z0, escape_dir, min_iou=iou_diff)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [r]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([vctm_curr, bbox_gen], [vctm_pred, adv_pred])
            if ious[0,0] + ious[1,1] > ious[0,1] + ious[1,0]:
                bbox_gen = vctm_pred
            adv_tracker.update(vctm_curr)
            vctm_tracker.update(bbox_gen)
        else:
            optimize_bbox(adv_tracker, vctm_tracker, z0, vctm_pred, lr=1, iteration=50, threshold=threshold)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [r]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([vctm_curr, bbox_gen], [vctm_pred, adv_pred])
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0]:
                adv_tracker.update(vctm_curr)
                vctm_tracker.update(bbox_gen)
                count += 1
            else:
                adv_tracker.update(bbox_gen)
                vctm_tracker.update(vctm_curr)

        z_temp = tf.identity(z0)
        traj_gen.append(z_temp)
    
    return convert_x_to_bbox_batch(traj_gen, aspect_ratio), switched

def get_image_point_tf(loc, K, w2c):
    """
    Input:
    - loc: 3 * 1 np.array([x, y, z])
    - K: intrinsic matrix
    - w2c: extrinsic matrix
    """
    # Calculate 2D projection of 3D coordinate

    # transform to camera coordinates
    point = tf.pad(loc, tf.constant([[0,1]]), constant_values=1.0)
    #print(point)
    point_camera = tf.matmul(w2c, tf.reshape(point, [4,1]))

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = tf.matmul(K, tf.reshape(point_camera, [3,1]))
    z = point_img[2]
    point_img = point_img / z
    # normalize
    #x = (point_img[0]/point_img[2])
    #y = (point_img[1]/point_img[2])

    return point_img[:2]

def get_image_point_batch_tf(loc, K, w2c):
    """
    Input:
    - loc: n * 3 * 1 np.array([x, y, z])
    - K: intrinsic matrix
    - w2c: extrinsic matrix
    """
    # Calculate 2D projection of 3D coordinate

    # transform to camera coordinates
    point = tf.pad(loc, tf.constant([[0,0],[0,1]]), constant_values=1.0)
    
    #print(point)
    point_camera = tf.matmul(w2c, tf.reshape(point, [loc.shape[0],4,1]))

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    #point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    permutation = tf.constant([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]], dtype=tf.float32)
    point_camera = tf.matmul(permutation, point_camera)[...,:3,0]

    # now project 3D->2D using the camera matrix
    point_img = tf.matmul(K, point_camera, transpose_b=True)
    z = point_img[2]
    point_img = point_img / z
    # normalize
    #x = (point_img[0]/point_img[2])
    #y = (point_img[1]/point_img[2])

    return tf.transpose(point_img[:2])

"""
def get_2d_bbox_tf(verts, proj_mat, w2c):
    x_max = -10000
    x_min = 10000
    y_max = -10000
    y_min = 10000
    for vert in verts:
        p = get_image_point_tf(vert, proj_mat, w2c)
        # Find the rightmost vertex
        if p[0] > x_max:
            x_max = p[0]
        # Find the leftmost vertex
        if p[0] < x_min:
            x_min = p[0]
        # Find the highest vertex
        if p[1] > y_max:
            y_max = p[1]
        # Find the lowest  vertex
        if p[1] < y_min:
            y_min = p[1]
    return tf.concat([x_min, y_min, x_max, y_max], 0)
"""

def get_2d_bbox_tf(verts, proj_mat, w2c):
    #verts_xy = tf.Variable([get_image_point_tf(vert, proj_mat, w2c) for vert in verts])
    verts_xy = get_image_point_batch_tf(verts, proj_mat, w2c)
    x_min = tf.reduce_min(verts_xy[:,0])
    x_max = tf.reduce_max(verts_xy[:,0])
    y_min = tf.reduce_min(verts_xy[:,1])
    y_max = tf.reduce_max(verts_xy[:,1])
    return tf.stack([x_min, y_min, x_max, y_max])


def format_location(verts):
    """
    Format Carla location to tensors
    """
    return tf.Variable([[vert.x, vert.y, vert.z] for vert in verts])

def shift_verts(verts, delta):
    #print(verts)
    shift = tf.pad(delta, tf.constant([[0,1]]))
    shift = tf.reshape(shift, [1,3])
    #shifts = tf.Variable([shift, shift, shift, shift, shift, shift, shift, shift])
    #shifts = tf.reshape(shifts, [8,3])
    shifts = tf.repeat(shift, [8], 0)
    #for vert in verts:
    #    vert += shift
        #vert += delta
    #print(shifts)
    #print(verts.shape)
    verts = verts + shifts
    return verts


def convert_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    #return tf.convert_to_tensor(np.array([x, y, s, r]).reshape((4, 1)), dtype=tf.float32)
    #return tf.reshape(tf.Variable([[x],[y],[s],[r]], dtype=tf.float32), (4,1))
    return tf.stack([x,y,s,r])

# Upgrade the optimization algorithm to 3D
def optimize_3d_coord(tracker_A, tracker_B, verts_A, bbox_B, proj_mat, w2c, lr=0.1, iteration=30, delta_location=0.3):
    """
    Perform gradient descent on the 3D coordinates of the attacker w.r.t. to the adversarial loss
    Input:
    - tracker_A: KalmanFilterTracker object for the attacker
    - tracker_B: KalmanFilterTracker object for the victim
    - verts_A: attacker's 3D bounding box vertices (8 corner representation)
    - bbox_B: victim's current KF predicted states in the form of bbox
    - lr: learning rate alpha
    - iteration: max number of iterations
    - delta_location: limit the relative intra-frame movement for the attacker, ensuring physical realizability
    Output:
    - A (x,y,z) point in Carla World that represents where the attacker should move (center displacement)
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    center_displacement = tf.Variable([0.,0.])
    #center_displacement = tf.Variable([1,1,0.1])
    z_B_next_pred = update_predict(tracker_B, convert_bbox_to_z(bbox_B)) # Victim's KF prediction for the next time step
    bbox_B_curr_pred = tracker_B.predict_no_trace()[0]
    verts_A_tensor = format_location(verts_A)
    #print(verts_A_tensor)
    #verts_A_shifted = shift_verts(verts_A_tensor, center_displacement)
    #print(verts_A_shifted)
    #g = -1
    for it in range(iteration):
        # Stop if the displacement is too large
        if tf.sqrt(center_displacement[0]**2 + center_displacement[1]**2) >= delta_location:
            break
        with tf.GradientTape() as tape:
            tape.watch(center_displacement)
            #print(verts_A_tensor)
            verts_A_shifted = shift_verts(verts_A_tensor, center_displacement)
            #print(verts_A_shifted)
            bbox_A = get_2d_bbox_tf(verts_A_shifted, proj_mat, w2c)
            #print(bbox_A)
            z_A = convert_to_z(bbox_A) #Problem!!!
            #print(z_A)
            z_predicted = update_predict(tracker_A, z_A)
            #print(z_predicted)
            bbox_A_predicted = convert_to_bbox(z_predicted) # Print and check
            iou_loss_1 = iou(bbox_A, bbox_B_curr_pred)
            iou_loss_2 = iou(bbox_A_predicted, convert_to_bbox(z_B_next_pred))
            loss = -(iou_loss_1 + iou_loss_2) + (distance_loss(bbox_A, bbox_B_curr_pred) + distance_loss_x(z_predicted, z_B_next_pred))
            # Change distance loss to non-x
            #loss = tf.reduce_sum(bbox_A_predicted)
            #print(loss)
        
        grads = tape.gradient(loss, [center_displacement])
        #g = grads
        optimizer.apply_gradients(zip(grads, [center_displacement]))

    return center_displacement.numpy()

