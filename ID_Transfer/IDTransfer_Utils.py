import numpy as np
import tensorflow as tf
import cv2
import math
from sort_TF import *
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm
#from yolov3.utils import pred_bbox

np_config.enable_numpy_behavior()

X_MARGIN, Y_MARGIN = 1.5, 1.5
SCALE_MARGIN, RATIO_MARGIN = 10, 0.25
VNAME = 'TEST_two_track.avi'
HEIGHT = 540
WIDTH = 960
CHANNELS = 3
BACKGROUND = np.full((HEIGHT,WIDTH,CHANNELS), 105, dtype=np.uint8)
S, R = 2400, 1.5
font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

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

def generate_move(start=(0,0), end=(0,0), time=None):
    """
    Return a sequence of (x,y) ordinates that represent a path from start to end
    """
    if time == None:
        time = min(abs(np.array(end) - np.array(start)))
    x_seq = np.linspace(start[0], end[0], time)
    y_seq = np.linspace(start[1], end[1], time)
    return np.column_stack((x_seq,y_seq))

def generate_bbox_with_noise(move, scale=2400, ratio=1.5, random_seed=0):
    """Return a series of bboxes according to specifie
     center moves, scale and ratio
     in the form of a list of [x1,y1,x2,y2] with added noise to simulate the detection result"""
    rg = np.random.default_rng(random_seed)
    bboxes = []
    for (x,y) in move:
        x += rg.normal(0, X_MARGIN)
        y += rg.normal(0, Y_MARGIN)
        scale += rg.normal(0, SCALE_MARGIN)
        #ratio += rg.normal(0, RATIO_MARGIN)
        bbox = convert_x_to_bbox([x,y,scale,ratio])[0]
        bboxes.append(bbox)
    return tf.Variable(bboxes)

def generate_bbox(move, scale=2400, ratio=1.5):
    """Return a Tensor of series of bboxes according to specifie
     center moves, scale and ratio
     in the form of a list of [x1,y1,x2,y2] to represent the ground truth"""
    bboxes = []
    for (x,y) in move:
        bbox = convert_x_to_bbox([x,y,scale,ratio])[0]
        bboxes.append(bbox)
    return tf.Variable(bboxes)

def jitter_bbox(bboxes, x_margin=X_MARGIN, y_margin=Y_MARGIN, scale_margin=SCALE_MARGIN, random_seed=0):
    rg = np.random.default_rng(random_seed)
    out_bboxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        x_jitter, y_jitter, s_jitter = rg.normal(0, x_margin), rg.normal(0, y_margin), rg.normal(0, scale_margin)
        x += x_jitter
        y += y_jitter
        s += s_jitter
        bbox = convert_x_to_bbox([x,y,s,r])[0]
        out_bboxes.append(bbox)
    return tf.Variable(out_bboxes, dtype=tf.float32)
    

def join_bboxes(*args):
    """Input different series of bbounding boxes, return an array them with ID attached to each path"""
    out_bboxes = np.expand_dims(np.insert(args[0], 4, 0, axis=1), axis=1)
    i = 1
    for bbox in args[1:]:
        bbox = np.expand_dims(np.insert(bbox, 4, i, axis=1), axis=1)
        out_bboxes = np.concatenate((out_bboxes, bbox), axis=1)
        i += 1
    return out_bboxes

def plot_sort_bbox(frames, attacking_frame=None, vname=VNAME, include_center_ids=None, true_color=(255,0,0), prior_color=(0,255,0), post_color=(0,0,255)):
    img = BACKGROUND.copy()
    tracker = Sort()
    vidout = cv2.VideoWriter(vname, 
        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (960, 540))
    curr_frame = 0
    
    for bboxes in frames:
        cv_im = img.copy()
        i = 0 # For plotting neatly
        prior = []
        
        for trk in tracker.trackers: # Prior estimated bbox
            states = trk.kf.x
            P = trk.kf.P
            bbox = trk.predict_no_trace()[0]
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            prior.append(bbox)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), prior_color, 1)
            cv2.putText(cv_im, "Velocity Est {}:{}".format(i, states[4:6]), (10, 70+i*20), font, font_scale, color=(0,0,0), thickness=1)
            i += 1
            #plot_cov_ellipse(cv_im, P[4:6, 4:6], (int(x),int(y)))
        

        for bbox in bboxes:
            bbox = bbox.astype(int)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), true_color, 1)
            objid = 'OBJ:{}'.format(int(bbox[4]))
            cv2.putText(cv_im, str(objid), (int(bbox[0]), int(bbox[1])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
            if bbox[4] in include_center_ids:
                x,y,_,_ = convert_bbox_to_z(bbox[:4])
                x, y = int(x), int(y)
                cv2.circle(cv_im, (x,y), 1, (0,0,0))
                if abs(x) < WIDTH and abs(y) < HEIGHT:
                    img[y,x,:] = (0,0,0)
        
        if len(prior) == bboxes.shape[0]:
            ious = iou_batch(prior, bboxes)
            iou_AA_BB = ious[0,0] + ious[1,1]
            iou_AB_BA = ious[0,1] + ious[1,0]
            cv2.putText(cv_im, "IoU(PA,TA)+IoU(PA,TA):{}".format(iou_AA_BB), (10,50), font, font_scale, color=(0,0,0), thickness=1)
            cv2.putText(cv_im, "IoU(PA,TB)+IoU(PB,TA):{}".format(iou_AB_BA), (10,30), font, font_scale, color=(0,0,0), thickness=1)

        posterior = tracker.update(bboxes[:,:4])
        for bbox in posterior:
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            x, y = int(x), int(y)
            trkid = int(bbox[4])
            P = tracker.get_P(trkid) # Estimated covariance
            # Plot the confidence ellipse for the center
            #plot_cov_ellipse(cv_im, P[:2,:2], (x,y))
            #cv2.circle(cv_im, (x,y), 1, (0,0,0)) # Point estimate of the predicted center
            #cv_im[x-1:x+2,y-1:y+2] = [0,0,0] # Point estimate of the predicted center
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), post_color, 1)
            cv2.putText(cv_im, str('SORT:{}'.format(trkid)), (int(bbox[2]), int(bbox[3])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Frame: {}".format(curr_frame), (WIDTH-200,20), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Detected", (WIDTH-200, HEIGHT-70), font, fontScale=font_scale, color=true_color, thickness=1)
        cv2.putText(cv_im, "KF Post Est", (WIDTH-200, HEIGHT-50), font, fontScale=font_scale, color=post_color, thickness=1)
        cv2.putText(cv_im, "KF Prior Est", (WIDTH-200, HEIGHT-30), font, fontScale=font_scale, color=prior_color, thickness=1)
        if curr_frame == attacking_frame:
            cv2.putText(cv_im, "ATTACKING", (0, 35), font, fontScale=2, color=(0,0,0), thickness=2)
        curr_frame += 1
        vidout.write(cv_im)
    vidout.release()

def plot_sort_bbox_w_bkgrd(frames, dir=None, vname=VNAME, include_objid=False, include_center_ids=[], true_color=(255,0,0), prior_color=(0,255,0), post_color=(0,0,255)):
    np_config.enable_numpy_behavior()
    tracker = Sort()
    vidout = cv2.VideoWriter(vname, 
        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (960, 540))
    curr_frame = 0
    
    for f, bboxes in enumerate(frames):
        cv_im = cv2.imread(dir+'\\{}.png'.format(f))
        i = 0 # For plotting neatly
        prior = []
        
        for trk in tracker.trackers: # Prior estimated bbox
            states = trk.kf.x
            P = trk.kf.P
            bbox = trk.predict_no_trace()[0]
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            prior.append(bbox)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), prior_color, 1)
            cv2.putText(cv_im, "Velocity Est {}:{}".format(i, states[4:6]), (10, 70+i*20), font, font_scale, color=(0,0,0), thickness=1)
            i += 1
            #plot_cov_ellipse(cv_im, P[4:6, 4:6], (int(x),int(y)))
        

        for bbox in bboxes:
            bbox = bbox.astype(int)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), true_color, 1)
            if include_objid:
                objid = 'OBJ:{}'.format(int(bbox[4]))
                cv2.putText(cv_im, str(objid), (int(bbox[0]), int(bbox[1])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
                if bbox[4] in include_center_ids:
                    x,y,_,_ = convert_bbox_to_z(bbox[:4])
                    x, y = int(x), int(y)
                    cv2.circle(cv_im, (x,y), 1, (0,0,0))
                    if abs(x) < WIDTH and abs(y) < HEIGHT:
                        cv_im[y,x,:] = (0,0,0)
        
        if len(prior) == bboxes.shape[0]:
            ious = iou_batch(prior, bboxes)
            iou_AA_BB = ious[0,0] + ious[1,1]
            iou_AB_BA = ious[0,1] + ious[1,0]
            cv2.putText(cv_im, "IoU(PA,TA)+IoU(PA,TA):{}".format(iou_AA_BB), (10,50), font, font_scale, color=(0,0,0), thickness=1)
            cv2.putText(cv_im, "IoU(PA,TB)+IoU(PB,TA):{}".format(iou_AB_BA), (10,30), font, font_scale, color=(0,0,0), thickness=1)

        posterior = tracker.update(bboxes[:,:4])
        for bbox in posterior:
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            x, y = int(x), int(y)
            trkid = int(bbox[4])
            P = tracker.get_P(trkid) # Estimated covariance
            # Plot the confidence ellipse for the center
            #plot_cov_ellipse(cv_im, P[:2,:2], (x,y))
            #cv2.circle(cv_im, (x,y), 1, (0,0,0)) # Point estimate of the predicted center
            #cv_im[x-1:x+2,y-1:y+2] = [0,0,0] # Point estimate of the predicted center
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), post_color, 1)
            cv2.putText(cv_im, str('SORT:{}'.format(trkid)), (int(bbox[2]), int(bbox[3])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Frame: {}".format(curr_frame), (WIDTH-200,20), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Detected", (WIDTH-200, HEIGHT-70), font, fontScale=font_scale, color=true_color, thickness=1)
        cv2.putText(cv_im, "KF Post Est", (WIDTH-200, HEIGHT-50), font, fontScale=font_scale, color=post_color, thickness=1)
        cv2.putText(cv_im, "KF Prior Est", (WIDTH-200, HEIGHT-30), font, fontScale=font_scale, color=prior_color, thickness=1)
        curr_frame += 1
        vidout.write(cv_im)
    vidout.release()

def update_predict(tracker, z, s=2400, r=1.5):
    if z.shape[0] == 2:
        z = tf.concat([z, [[s],[r]]], axis=0)
    # Update step
    dim_x, dim_z = tracker.kf.dim_x, tracker.kf.dim_z
    x = tracker.kf.x
    y = tracker.kf.y
    R = tracker.kf.R
    H = tf.constant([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=tf.float32)
    P = tracker.kf.P
    y = z - tf.matmul(H, x)
    PHT = tf.matmul(P, tf.transpose(H))
    S = tf.matmul(H, PHT) + R
    SI = tf.linalg.inv(S)
    K = tf.matmul(PHT, SI)
    x = x + tf.matmul(K, y)
    I_KH = tf.eye(dim_x) - tf.matmul(K, H)
    P = tf.matmul(tf.matmul(I_KH, P), tf.transpose(I_KH)) + tf.matmul(tf.matmul(K, R), tf.transpose(K))
    #x_post = tf.identity(x)
    #P_post = tf.identity(P)
    # Prediction Step
    F = tf.constant([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], dtype=tf.float32)
    x = tf.matmul(F, x)
    return x

def convert_to_bbox(x):
    s, r = x[2], x[3]
    w = tf.sqrt(s * r)
    h = s / w
    return tf.Variable([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.], dtype=tf.float32)

def optimize_bbox_A(tracker_A, z_A, bbox_B, bbox_B1, lr=0.1, iteration=30, threshold=0.9, s=2400, r=1.5):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    z_prev = tf.identity(z_A)
    for it in range(iteration):
        if iou(convert_to_bbox(tf.concat([z_A, [[s],[r]]], axis=0)), convert_to_bbox(tf.concat([z_prev, [[s],[r]]], axis=0))) <= threshold:
                break
        with tf.GradientTape() as tape:
            adv_z = update_predict(tracker_A, z_A)
            z_A_temp = tf.concat([z_A, [[s],[r]]], axis=0)
            w = tf.sqrt(adv_z[2] * adv_z[3])
            h = adv_z[2] / w
            iou_loss_1 = iou([adv_z[0]-w/2.,adv_z[1]-h/2.,adv_z[0]+w/2.,adv_z[1]+h/2.], bbox_B1)
            iou_loss_2 = iou([z_A_temp[0]-w/2.,z_A_temp[1]-h/2.,z_A_temp[0]+w/2.,z_A_temp[1]+h/2.], bbox_B)
            #v = aspect_ratio_loss_x(adv_z, convert_bbox_to_z(bbox_B))
            #alpha = v / (1 - iou_loss + v)
            loss = -iou_loss_1 -iou_loss_2 + distance_loss_x(adv_z, convert_bbox_to_z(bbox_B1)) + distance_loss_x(z_A_temp, convert_bbox_to_z(bbox_B)) #+ alpha * v
        grads = tape.gradient(loss, [z_A])
        opt.apply_gradients(zip(grads, [z_A]))
    return z_A

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

def convert_x_to_bbox_batch(xs, s=S, r=R):
    bbox_out = []
    for x in xs:
        bbox_out.append(convert_to_bbox(tf.concat([x, [[s],[r]]], axis=0)))
    return np.squeeze(np.array(bbox_out))

def get_s_r_batch(bboxes):
    out = []
    for bbox in bboxes:
        _,_,s,r = convert_bbox_to_z(bbox)
        out.append((s,r))
    return out


def trajGen(adv_init, vctm_traj, escape_dir=(0,0), threshold=0.9, patience=0, s=2400, r=1.5):
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
    #_,_,s,r = adv_z
    #s, r = int(s), int(r)
    z0 = tf.Variable(adv_z[:2])
    traj_gen = [tf.identity(z0)]
    adv_tracker = KalmanBoxTracker(adv_init)
    vctm_tracker = KalmanBoxTracker(vctm_traj[0])
    count = 0
    stop_pos = None
    for i in range(1, length):
        vctm_curr = vctm_traj[i]
        vctm_pred = vctm_tracker.predict()[0]
        adv_pred = adv_tracker.predict_no_trace()[0]
        adv_curr = convert_x_to_bbox(tf.concat([traj_gen[-1], [[s], [r]]], axis=0))[0]
        ious = iou_batch([vctm_curr, adv_curr], [vctm_pred, adv_pred])
        print(ious)
        if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] and count >= patience:
            if escape_dir == (0,0):
                if stop_pos is None:
                    z0 = tf.Variable(convert_bbox_to_z(vctm_pred)[:2])
                    stop_pos = tf.Variable(tf.identity(z0))
                else:
                    z0 = stop_pos
            else:
                iou_diff = max(ious[0,0], 0.95)
                move_covertly(z0, escape_dir, min_iou=iou_diff)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s], [r]]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([vctm_curr, bbox_gen], [vctm_pred, adv_pred])
            if ious[0,0] + ious[1,1] > ious[0,1] + ious[1,0]:
                bbox_gen = vctm_pred
            adv_tracker.update(vctm_curr)
            vctm_tracker.update(bbox_gen)
        else:
            optimize_bbox_A(adv_tracker, z0, vctm_pred, vctm_pred, lr=0.5, iteration=50, threshold=threshold, s=s, r=r)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s], [r]]], axis=0))[0]
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
    
    return convert_x_to_bbox_batch(traj_gen, s, r)

def trajGen_s(adv_init, vctm_traj, escape_dir=(0,0), threshold=0.9, patience=0, iteration=50):
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
    #_,_,s,r = adv_z
    #s, r = int(s), int(r)
    z0 = tf.Variable(adv_z[:2])
    #traj_gen = [tf.identity(z0)]
    _,_,s0,r0 = convert_bbox_to_z(vctm_traj[0])
    s0, r0 = int(s0), int(r0)
    s1, r1 = s0, r0
    bbox_lst = [convert_x_to_bbox(tf.concat([tf.identity(z0), [[s0], [r0]]], axis=0))]
    adv_tracker = KalmanBoxTracker(bbox_lst[0].reshape(4))
    vctm_tracker = KalmanBoxTracker(vctm_traj[0])
    count = 0
    stop_pos = None
    for i in range(1, length):
        vctm_curr = vctm_traj[i]
        vctm_pred = vctm_tracker.predict()[0]
        try:
            _,_,s1_temp,r1_temp = convert_bbox_to_z(vctm_pred)
            s1, r1 = int(s1_temp), int(r1_temp)
        except:
            pass
        adv_pred = adv_tracker.predict_no_trace()[0]
        adv_curr = bbox_lst[-1].reshape(4)
        #adv_curr = convert_x_to_bbox(tf.concat([traj_gen[-1], [[s0], [r0]]], axis=0))[0]
        ious = iou_batch([vctm_curr, adv_curr], [vctm_pred, adv_pred])
        print(adv_pred)
        print(vctm_pred)
        print(ious)
        if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] and count >= patience:
            if escape_dir == (0,0):
                if stop_pos is None:
                    z = convert_bbox_to_z(vctm_pred)
                    _,_,s0,r0 = z
                    s0,r0 = int(s0), int(r0)
                    z0 = tf.Variable(z[:2])
                    stop_pos = tf.Variable(tf.identity(z0))
                bbox_gen = convert_x_to_bbox(tf.concat([stop_pos, [[s0], [r0]]], axis=0))[0]
            else:
                iou_diff = max(ious[0,0], 0.95)
                move_covertly(z0, escape_dir, min_iou=iou_diff)
                bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s1], [r1]]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([vctm_curr, bbox_gen], [vctm_pred, adv_pred])
            if ious[0,0] + ious[1,1] > ious[0,1] + ious[1,0]:
                bbox_gen = vctm_pred
            adv_tracker.update(vctm_curr)
            vctm_tracker.update(bbox_gen)
        else:
            #_,_,s1,r1 = convert_bbox_to_z(vctm_pred)
            #s1, r1 = int(s1), int(r1)
            optimize_bbox_A(adv_tracker, z0, vctm_pred, vctm_pred, lr=0.5, iteration=iteration, threshold=threshold, s=s1, r=r1)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s1], [r1]]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([vctm_curr, bbox_gen], [vctm_pred, adv_pred])
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0]:
                adv_tracker.update(vctm_curr)
                vctm_tracker.update(bbox_gen)
                count += 1
            else:
                adv_tracker.update(bbox_gen)
                vctm_tracker.update(vctm_curr)

        #z_temp = tf.identity(z0)
        #traj_gen.append(z_temp)
        #bbox_lst.append(convert_x_to_bbox(tf.concat([z_temp, [[s1], [r1]]], axis=0)))
        bbox_lst.append(bbox_gen)
        #s0,r0 = s1,r1
    
    return np.squeeze(np.array(bbox_lst))

def plot_yolo_sort_bbox(yolo, n_frames, dir=None, vname=VNAME, include_center_ids=[], true_color=(255,0,0), prior_color=(0,255,0), post_color=(0,0,255)):
    np_config.enable_numpy_behavior()
    tracker = Sort()
    vidout = cv2.VideoWriter(vname, 
        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (960, 540))
    
    for curr_frame in range(n_frames):
        cv_im = cv2.imread(dir+'\\{}.png'.format(curr_frame))
        bboxes = pred_bbox(yolo, cv_im)[:,:4]
        i = 0 # For plotting neatly
        prior = []
        
        for trk in tracker.trackers: # Prior estimated bbox
            states = trk.kf.x
            P = trk.kf.P
            bbox = trk.predict_no_trace()[0]
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            prior.append(bbox)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), prior_color, 1)
            cv2.putText(cv_im, "Velocity Est {}:{}".format(i, states[4:6]), (10, 70+i*20), font, font_scale, color=(0,0,0), thickness=1)
            i += 1
            #plot_cov_ellipse(cv_im, P[4:6, 4:6], (int(x),int(y)))
        

        for bbox in bboxes:
            bbox = bbox.astype(int)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), true_color, 1)
            #objid = 'OBJ:{}'.format(int(bbox[4]))
            #cv2.putText(cv_im, str(objid), (int(bbox[0]), int(bbox[1])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
            #if bbox[4] in include_center_ids:
            #    x,y,_,_ = convert_bbox_to_z(bbox[:4])
            #    x, y = int(x), int(y)
            #    cv2.circle(cv_im, (x,y), 1, (0,0,0))
            #    if abs(x) < WIDTH and abs(y) < HEIGHT:
            #        cv_im[y,x,:] = (0,0,0)
        
        if len(prior) == bboxes.shape[0]:
            ious = iou_batch(prior, bboxes)
            iou_AA_BB = ious[0,0] + ious[1,1]
            iou_AB_BA = ious[0,1] + ious[1,0]
            cv2.putText(cv_im, "IoU(PA,TA)+IoU(PA,TA):{}".format(iou_AA_BB), (10,50), font, font_scale, color=(0,0,0), thickness=1)
            cv2.putText(cv_im, "IoU(PA,TB)+IoU(PB,TA):{}".format(iou_AB_BA), (10,30), font, font_scale, color=(0,0,0), thickness=1)

        posterior = tracker.update(bboxes[:,:4])
        for bbox in posterior:
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            x, y = int(x), int(y)
            trkid = int(bbox[4])
            P = tracker.get_P(trkid) # Estimated covariance
            # Plot the confidence ellipse for the center
            #plot_cov_ellipse(cv_im, P[:2,:2], (x,y))
            #cv2.circle(cv_im, (x,y), 1, (0,0,0)) # Point estimate of the predicted center
            #cv_im[x-1:x+2,y-1:y+2] = [0,0,0] # Point estimate of the predicted center
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), post_color, 1)
            cv2.putText(cv_im, str('SORT:{}'.format(trkid)), (int(bbox[2]), int(bbox[3])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Frame: {}".format(curr_frame), (WIDTH-200,20), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Detected", (WIDTH-200, HEIGHT-70), font, fontScale=font_scale, color=true_color, thickness=1)
        cv2.putText(cv_im, "KF Post Est", (WIDTH-200, HEIGHT-50), font, fontScale=font_scale, color=post_color, thickness=1)
        cv2.putText(cv_im, "KF Prior Est", (WIDTH-200, HEIGHT-30), font, fontScale=font_scale, color=prior_color, thickness=1)

        vidout.write(cv_im)
    vidout.release()

def get_yolo_bbox(Yolo, dir, n_frames):
    bboxes = []
    for frame in range(n_frames):
        img = cv2.imread(dir+'\\{}.png'.format(frame))
        bbox = pred_bbox(Yolo, img)[...,:4]
        bboxes.append(bbox)
    return bboxes

def generate_traj(target_move, attacker_spawn_range=(0,960,0,360), s=2400, r=1.5, stop_pos=None, seed=None):
    """
    Input: 
        - Target_move: a sequence of the center of the target
        - Attacker_spawn_range: (x_start, x_end, y_start, y_end)
        - s,r: scale and ratio of the bboxes
    Output:
        - A sequence of the center of the attacker represent the adversarial trajectory
    """
    if seed is not None:
        np.random.seed(seed)
    target_bbox = generate_bbox_with_noise(target_move, scale=s, ratio=r)
    attacker_init_center = [np.random.uniform(attacker_spawn_range[0], attacker_spawn_range[1]),
                            np.random.uniform(attacker_spawn_range[2], attacker_spawn_range[3])]
    bbox_0 = convert_to_bbox(attacker_init_center + [s,r])
    z0 = tf.Variable(convert_bbox_to_z(bbox_0)[:2])
    adv_traj = [tf.identity(z0)]
    adv_tracker = KalmanBoxTracker(bbox_0)
    #print(target_bbox)
    target_tracker = KalmanBoxTracker(target_bbox[0])
    stop_pos = None
    switched = False

    for i in tqdm(range(1, target_bbox.shape[0])):
        target_curr = target_bbox[i]
        target_pred = target_tracker.predict()[0]
        adv_pred = adv_tracker.predict_no_trace()[0]
        adv_curr = convert_x_to_bbox(tf.concat([adv_traj[-1],[[s],[r]]], axis=0))[0]
        ious = iou_batch([target_curr, adv_curr], [target_pred, adv_pred])
        if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0]: #IDs are swapped
            switched = True
            if stop_pos is None:
                z0 = tf.Variable(convert_bbox_to_z(target_pred)[:2])
                stop_pos = tf.Variable(tf.identity(z0))
            else:
                z0 = stop_pos
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s],[r]]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([target_curr, bbox_gen], [target_pred, adv_pred])
            if ious[0,0] + ious[1,1] > ious[0,1] + ious[1,0]:
                bbox_gen = target_pred
            adv_tracker.update(target_curr)
            target_tracker.update(bbox_gen)
        else:
            optimize_bbox_A(adv_tracker, z0, target_pred, target_pred, lr=0.3, iteration=20, threshold=0.75, s=s, r=r)
            bbox_gen = convert_x_to_bbox(tf.concat([z0, [[s],[r]]], axis=0))[0]
            adv_pred = adv_tracker.predict()[0]
            ious = iou_batch([target_curr, bbox_gen], [target_pred, adv_pred])
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0]:
                adv_tracker.update(target_curr)
                adv_tracker.update(bbox_gen)
            else:
                adv_tracker.update(bbox_gen)
                target_tracker.update(target_curr)
        
        z_temp = tf.identity(z0)
        adv_traj.append(z_temp)
    
    return adv_traj if switched else None

def moving_average(data, window_size):
    """Compute the moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')