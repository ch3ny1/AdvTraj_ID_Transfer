import numpy as np
import carla
from carla import Transform, Location, Rotation


# Intrinsic matrix (cwang)
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# K is the intrinsic matrix, w2c is the extrinsic matrix (cwang)
def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def get_2d_bbox(vehicle, proj_mat, w2c):
    bb = vehicle.bounding_box
    verts = [v for v in bb.get_world_vertices(vehicle.get_transform())]
    x_max = -10000
    x_min = 10000
    y_max = -10000
    y_min = 10000
    for vert in verts:
        p = get_image_point(vert, proj_mat, w2c)
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
    return [x_min, y_min, x_max, y_max]

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

# cwang
# Project points in the image back to the world coordinates
def get_world_point(p, inv_proj_mat, c2w, camera_loc, target_z): 
    """
    Parameters:
        p: point on the image (x,y)
        inv_proj_mat: inverse of the intrinsic matrix of the camera (inverse of K)
        c2w: inverse of the extrinsic matrix (inverse of w2c)
        camera_loc: location of the camera
        target_z: for uniqueness of the world point
    """
    point_img = np.array([p[0], p[1], 1.0])
    point_camera = np.dot(inv_proj_mat, point_img)
    # New we must change from "standard" coordinate system to  UE4's
    # (y, -z, x) -> (x, y ,z)
    # and we add the fourth componebonent also
    point_camera = [point_camera[2], point_camera[0], -point_camera[1], 1]
    point_world = np.dot(c2w, point_camera)[:3]
    # Uniquely define the point according to given z
    camera_coord = [camera_loc.x, camera_loc.y, camera_loc.z]
    point_world = (point_world - camera_coord) * ((target_z - camera_loc.z) / (point_world[2] - camera_loc.z)) + camera_coord
    return Location(x=point_world[0], y=point_world[1], z=point_world[2])
    #return point_world

def get_2d_bbox_with_shift(vehicle, proj_mat, w2c, shift):
    """
    Get the 2D bbox of an actor as if it is moved with a shift (Vector3D) in the Carla world
    """
    bb = vehicle.bounding_box
    verts = [v + shift for v in bb.get_world_vertices(vehicle.get_transform())]
    x_max = -10000
    x_min = 10000
    y_max = -10000
    y_min = 10000
    for vert in verts:
        p = get_image_point(vert, proj_mat, w2c)
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
    return [x_min, y_min, x_max, y_max]

def get_verts(actor):
    bb = actor.bounding_box
    verts = [v for v in bb.get_world_vertices(actor.get_transform())]
    return verts

