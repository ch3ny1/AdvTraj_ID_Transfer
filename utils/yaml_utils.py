import re
import yaml
import os
import math
from carla import Transform, Location, Rotation, Vector3D, WalkerControl

import numpy as np


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_yaml_wo_overwriting(data, save_name):
    """
    Save the yaml file without overwriting the existing one.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """
    if os.path.exists(save_name):
        prev_data = load_yaml(save_name)
        data = {**data, **prev_data}

    save_yaml(data, save_name)

def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)

    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1)
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7
    )
    geometry_param = {
        'L1': L1,
        'L2': L2,
        'W1': W1,
        'W2': W2,
        'H1': H1,
        'H2': H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    param["model"]["args"]["geometry_param"] = geometry_param
    return param





"""
def carla_transform_parser(param):
    param['victim_settings']['spawn_transform'] = list_to_transform(param['victim_settings']['spawn_transform'])
    param['victim_settings']['end_transform'] = list_to_transform(param['victim_settings']['end_transform'])
    param['attacker_settings']['spawn_transform'] = list_to_transform(param['attacker_settings']['spawn_transform'])
    for (i, camera) in param['cameras'].items():
        param['cameras'][camera]['transform'] = list_to_transform(param['cameras'][camera]['transform'])
    return param
"""

def list_to_transform(lst):
    return Transform(Location(x = lst[0],
                              y = lst[1],
                              z = lst[2]),
                    Rotation(pitch = lst[3],
                             yaw = lst[4],
                             roll = lst[5]))

def list_to_walkercontrol(lst):
    return WalkerControl(direction=Vector3D(lst[0], lst[1], lst[2]), speed=lst[3])

def list_to_vector3D(lst):
    return Vector3D(lst[0], lst[1], lst[2])

def transform_to_list(t):
    location = t.location
    rotation = t.rotation
    return [
        location.x,
        location.y,
        location.z,
        rotation.pitch,
        rotation.yaw,
        rotation.roll
    ]