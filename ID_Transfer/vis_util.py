import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.python.ops.numpy_ops import np_config
from ID_Transfer.ID_Transfer import *
from matplotlib.patches import FancyArrowPatch 

def plot_sort_bbox_w_bkgrd(bboxes, image_dir, output_path, true_color=(255,0,0), post_color=(0,255,0),):
    """
    Plot attack results on Carla
    Input:
        - bboxes: np.array n_frame * n_object * 4 bboxes
        - frame_index: np.array n * 1 which frames to plot the results
        - image_dir: image directory of the BDD100K dataset
    """
    np_config.enable_numpy_behavior()
    tracker = Sort(iou_threshold=0.05)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 5.0, (960, 540))
    
    for i in range(bboxes.shape[0]):
        cv_im = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(i)))
        object_bboxes = bboxes[i]
        for bbox in object_bboxes: # Plot the GT bbox
            bbox = bbox.astype(int)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), true_color, 1)
        posterior = tracker.update(object_bboxes[:,:4]) # Get SORT predicted ID
        for bbox in posterior:
            x,y,_,_ = convert_bbox_to_z(bbox[:4])
            x, y = int(x), int(y)
            trkid = int(bbox[4])
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), post_color, 1)
            cv2.putText(cv_im, str('SORT:{}'.format(trkid)), (int(bbox[2]), int(bbox[3])), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,255,255), thickness=1)
        
        cv2.putText(cv_im, "Frame: {}".format(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Detected", (1280-200, 720-70), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=true_color, thickness=1)
        cv2.putText(cv_im, "KF Post Est", (1280-200, 720-50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=post_color, thickness=1)
        out.write(cv_im)
    out.release()

def plot_sort_bbox(frames, attacking_frame=None, output_path='test.avi', true_color=(255,0,0), prior_color=(0,255,0), post_color=(0,0,255)):
    np_config.enable_numpy_behavior()
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN
    img = np.full((540,960,3), 105, dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidout = cv2.VideoWriter(output_path, fourcc, 10.0, (960, 540))
    tracker = Sort()
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

        for bbox in bboxes:
            bbox = bbox.astype(int)
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), true_color, 1)
            objid = 'OBJ:{}'.format(int(bbox[4]))
            cv2.putText(cv_im, str(objid), (int(bbox[0]), int(bbox[1])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        
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
            cv2.rectangle(cv_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), post_color, 1)
            cv2.putText(cv_im, str('SORT:{}'.format(trkid)), (int(bbox[2]), int(bbox[3])), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Frame: {}".format(curr_frame), (960-200,20), font, fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(cv_im, "Detected", (960-200, 540-70), font, fontScale=font_scale, color=true_color, thickness=1)
        cv2.putText(cv_im, "KF Post Est", (960-200, 540-50), font, fontScale=font_scale, color=post_color, thickness=1)
        cv2.putText(cv_im, "KF Prior Est", (960-200, 540-30), font, fontScale=font_scale, color=prior_color, thickness=1)
        if curr_frame == attacking_frame:
            cv2.putText(cv_im, "ATTACKING", (0, 35), font, fontScale=2, color=(0,0,0), thickness=2)
        curr_frame += 1
        vidout.write(cv_im)
    vidout.release()

def calculate_speed(bboxes, center_only=True):
    if not center_only:
        speeds = np.zeros(shape=(bboxes.shape[0]-1, bboxes.shape[1], 1))
        for t, bbox in enumerate(bboxes):
            if t==0:
                pass
            for i, bb in enumerate(bbox):
                speed = (bb[2]+bb[0]) / 2 - (bboxes[t-1][i][2]+bboxes[t-1][i][0]) / 2
                speeds[t-1,i,0] = speed
    else:
        speeds = np.zeros(shape=(bboxes.shape[0], bboxes.shape[2]-1, 1))
        for i, centers in enumerate(bboxes):
            for t in range(centers.shape[1]):
                if t==0:
                    pass
                speed = math.sqrt((centers[0][t]-centers[0][t-1])**2 + (centers[1][t]-centers[1][t-1])**2)
                speeds[i, t-1, 0] = speed
                
    return speeds

def calculate_distance(bboxes, center_target=None, center_only=True):
    if not center_only:
        distance = np.zeros(shape=(bboxes.shape[0]-1))
        for t, bbox in enumerate(bboxes):
            if t==0:
                pass
            distance[t-1] = ((bbox[1,2]+bbox[1,2]) / 2) - ((bbox[0,2]+bbox[0,2]) / 2)
    else:
        distance = np.zeros(shape=(bboxes.shape[0], bboxes.shape[2]-1))
        for i, centers in enumerate(bboxes):
            for t in range(centers.shape[1]):
                if t==0:
                    pass
                distance[i,t-1] = math.sqrt((center_target[t,0]-bboxes[i,0,t])**2+(center_target[t,1]-bboxes[i,1,t])**2)
    return distance

def calculate_velocity(bboxes, center_only=True):
    if not center_only:
        speeds = np.zeros(shape=(bboxes.shape[0]-1, bboxes.shape[1], 1))
        for t, bbox in enumerate(bboxes):
            if t==0:
                pass
            for i, bb in enumerate(bbox):
                speed = (bb[2]+bb[0]) / 2 - (bboxes[t-1][i][2]+bboxes[t-1][i][0]) / 2
                speeds[t-1,i,0] = speed
    else:
        speeds = np.zeros(shape=(bboxes.shape[0], bboxes.shape[2]-1, 2))
        for i, centers in enumerate(bboxes):
            for t in range(centers.shape[1]):
                if t==0:
                    pass
                #speed = math.sqrt((centers[0][t]-centers[0][t-1])**2 + (centers[1][t]-centers[1][t-1])**2)
                speeds[i, t-1, 0] = centers[0][t]-centers[0][t-1]
                speeds[i, t-1, 1] = centers[1][t]-centers[1][t-1]
                
    return speeds

def moving_average(data, window_size):
    """Compute the moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def arrow(x,y,ax,n):
    d = len(x)//(n+1)    
    ind = np.arange(d,len(x),d)
    for i in ind:
        ar = FancyArrowPatch ((x[i-1],y[i-1]),(x[i],y[i]), 
                              arrowstyle='->', mutation_scale=10, alpha=0.5, zorder=0)
        ax.add_patch(ar)