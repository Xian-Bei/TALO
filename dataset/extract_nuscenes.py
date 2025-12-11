from functools import partial
import sys
import time
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import json
import multiprocessing
import shutil
import ipdb
import os
import numpy as np
from nuscenes.nuscenes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from PIL import Image

w = 1.73
l = 4.08
h = 1.84
eps = 0.1

def extract(scene, data_root, nusc, save_root):
    # print('Processing ', scene['name'])
    cam_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    scene_dir = os.path.join(save_root, scene['name'])
    shutil.rmtree(scene_dir, ignore_errors=True)
    ######################################################################
    for dir_name in ['lidar', 'ego', 'intrinsic']:
        dir_path = os.path.join(scene_dir, dir_name)
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)
    for cam_name in cam_list:
        cam_dir = os.path.join(scene_dir, 'image', cam_name)
        shutil.rmtree(cam_dir, ignore_errors=True)
        os.makedirs(cam_dir, exist_ok=True)
        cam2world_dir = os.path.join(scene_dir, 'cam2world', cam_name)
        shutil.rmtree(cam2world_dir, ignore_errors=True)
        os.makedirs(cam2world_dir, exist_ok=True)
    ######################################################################
    first_sample_token = scene['first_sample_token']
    key_frame_id = 0
    sample_token = first_sample_token
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        # print('Processing ', key_frame_id)

        sensor_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', sensor_token)
        lidar2ego = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar2ego = transform_matrix(lidar2ego['translation'], Quaternion(lidar2ego['rotation']), inverse=False)
        ego2world = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego2world = transform_matrix(ego2world['translation'], Quaternion(ego2world['rotation']), inverse=False)
        lidar2world = ego2world @ lidar2ego
        # ######################################################################
        lidar_path = os.path.join(data_root, lidar_data['filename'])
        lidar_points = LidarPointCloud.from_file(lidar_path)
        lidar_points.transform(lidar2world)
        lidar_save_path = os.path.join(scene_dir, 'lidar', f'{key_frame_id:0>3d}.bin')
        xyz_lidar = lidar_points.points.T.astype(np.float32).reshape(-1, 4)[:, :3]
        inside_ego_range = np.array([
            [-w/2-eps, -l*0.4-eps, -h-eps],
            [w/2+eps, l*0.6+eps, eps]
        ])
        inside_mask = (xyz_lidar >= inside_ego_range[0:1]).all(1) & (xyz_lidar <= inside_ego_range[1:]).all(1) 
        xyz_lidar = xyz_lidar[~inside_mask]
        xyz_lidar.tofile(lidar_save_path)
        
        for sensor_name in cam_list:
            sensor_token = sample['data'][sensor_name]
            cam_data = nusc.get('sample_data', sensor_token)
            ######################################################################
            key_path = nusc.get_sample_data_path(cam_data['token'])
            key_save_path = os.path.join(scene_dir, 'image', sensor_name, f'{key_frame_id:0>3d}.jpg')
            shutil.copy(key_path, key_save_path)
            ######################################################################
            camera_pose = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam2ego = transform_matrix(camera_pose['translation'], Quaternion(camera_pose['rotation']), inverse=False)
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego2world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
            cam2world = ego2world @ cam2ego
            cam2world_save_path = os.path.join(scene_dir, 'cam2world', sensor_name, f'{key_frame_id:0>3d}.txt')
            np.savetxt(cam2world_save_path, cam2world)
            ######################################################################
            intrinsics = np.array(camera_pose['camera_intrinsic'])
            np.savetxt(os.path.join(scene_dir, 'intrinsic', f'{sensor_name}.txt'), intrinsics)
        key_frame_id += 1
                
        sample_token = sample['next']
    print('Processing done ', scene['name'])

def thread_worker(scene, data_root, nusc, save_root):
    result = extract(scene, data_root=data_root, nusc=nusc, save_root=save_root)
    return scene['name'], result 
    
if __name__ == "__main__":
    data_root = '/media/fengyi/bb/nuscenes'
    save_root = f'Data/nuscenes'

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    scene_name_list = [
        "scene-0003",
        "scene-0012",
        "scene-0013",
        "scene-0036",
        "scene-0039",
        "scene-0092", 
        "scene-0094", 
    ]
    scene_list = [scene for scene in nusc.scene if scene['name'] in scene_name_list]

    for scene in scene_list:
        extract(scene, data_root=data_root, nusc=nusc, save_root=save_root)

