'''This code is adapted from DriveStudio (https://github.com/ziyc/drivestudio/blob/main/datasets/waymo/)'''
# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import argparse
from glob import glob
import json
import os
import shutil

import ipdb
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from os.path import join


WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
# TODO(ziyu): consider all dynamic classes
WAYMO_DYNAMIC_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
WAYMO_HUMAN_CLASSES = ['Pedestrian', 'Cyclist']
WAYMO_VEHICLE_CLASSES = ['Vehicle']

OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(
            range_image_origins, dtype=range_image_polar_dtype
        )
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian



class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "dynamic_masks",
        ],
        workers=64,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            "FRONT",
            "FRONT_LEFT",
            "FRONT_RIGHT",
            "SIDE_LEFT",
            "SIDE_RIGHT",
        ]
        self.lidar_list = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        # a list of tfrecord pathnames
        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))
        self.scene_name = [f"{name.split('-')[1].split('_')[0]}" for name in sorted(os.listdir(self.load_dir)) if name.endswith('.tfrecord')]
        

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        self.create_folder(file_idx)
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames = sum(1 for _ in dataset)
        frame_idx = 0
        for i, data in enumerate(
            tqdm(dataset, desc=f"File {self.scene_name[file_idx]}", total=num_frames, dynamic_ncols=True)
        ):
            if i % 5 != 0:
                continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (
                self.selected_waymo_locations is not None
                and frame.context.stats.location not in self.selected_waymo_locations
            ):
                continue
            if "images" in self.process_keys:
                self.save_image(frame, self.scene_name[file_idx], frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(frame, self.scene_name[file_idx], frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(frame, self.scene_name[file_idx], frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(frame, self.scene_name[file_idx], frame_idx, class_valid='all')
            frame_idx += 1

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)


    def save_image(self, frame, file_name, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = (
                join(self.save_dir, f"{file_name}/image/{self.cam_list[img.name - 1]}/{str(frame_idx).zfill(3)}.jpg")
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_name, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_name (str): Current file name.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        
        ego2world = np.array(frame.pose.transform).reshape(4, 4)

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            cam2ego = np.array(camera.extrinsic.transform).reshape(4, 4) @ OPENCV2DATASET
            cam2world = ego2world @ cam2ego
            intrinsic = list(camera.intrinsic)

            def _K_from_intrinsic_list(intrinsic_list) -> np.ndarray:
                fx, fy, cx, cy = intrinsic_list[:4]
                K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float64)
                return K
            
            intrinsic = _K_from_intrinsic_list(intrinsic)
            np.savetxt(
                join(f"{self.save_dir}", f"{file_name}/cam2world/{self.cam_list[camera.name - 1]}/{str(frame_idx).zfill(3)}.txt"),
                cam2world,
            )
            np.savetxt(
                join(f"{self.save_dir}", f"{file_name}/intrinsic/{self.cam_list[camera.name - 1]}.txt"),
                intrinsic,
            )
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto


    def save_lidar(self, frame, file_name, frame_idx):
        """Save lidar point cloud.
        Args:
            color (bool): Whether to colorize lidar points using camera projection.
                        If False, only XYZ (float32) will be saved.
        """
        (range_images, camera_projections, seg_labels, range_image_top_pose) = \
            parse_range_image_and_camera_projection(frame)
        if range_image_top_pose is None:
            return

        frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims
        )
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2],
        )
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
        )

        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        all_pts, all_rgb = [], []

        for c in calibrations:
            ri_index = 0
            if c.name not in range_images:
                continue
            range_image = range_images[c.name][ri_index]

            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0],
                )
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)
            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims
            )
            range_image_mask = range_image_tensor[..., 0] > 0
            mask_index = tf.where(range_image_mask)

            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = tf.expand_dims(range_image_top_pose_tensor, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)

            _, points_cart = extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local,
            )
            points_cart = tf.squeeze(points_cart, axis=0)
            points_tensor = tf.gather_nd(points_cart, mask_index)
            pts = points_tensor.numpy()
            all_pts.append(pts)

        pts = np.concatenate(all_pts, axis=0)

        ego2world = np.array(frame.pose.transform).reshape(4, 4)
        pts = (ego2world[:3, :3] @ pts.T + ego2world[:3, 3:4]).T
        pts.astype(np.float32).tofile(
            f"{self.save_dir}/{file_name}/lidar/{str(frame_idx).zfill(3)}.bin"
        )

    def save_dynamic_mask(self, frame, file_name, frame_idx, class_valid='all'):
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = WAYMO_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = WAYMO_HUMAN_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = WAYMO_VEHICLE_CLASSES
        mask_dir = f"{self.save_dir}/{file_name}/dynamic_mask"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            # dynamic_mask
            img_path = (
                join(self.save_dir, f"{file_name}/image/{self.cam_list[img.name - 1]}/{str(frame_idx).zfill(3)}.jpg")
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            filter_available = any(
                [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
            )
            calibration = next(
                cc for cc in frame.context.camera_calibrations if cc.name == img.name
            )
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                
                class_name = WAYMO_CLASSES[label.type]
                if class_name not in VALID_CLASSES:
                    continue

                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[
                    0
                ].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(
                    frame.pose, calibration, corners
                )
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = os.path.join(mask_dir, f"{self.cam_list[img.name - 1]}/{str(frame_idx).zfill(3)}.png")
            dynamic_mask.save(dynamic_mask_path)

    def create_folder(self, i):
        """Create folder for data preprocessing."""
        shutil.rmtree(join(self.save_dir, f"{self.scene_name[i]}"), ignore_errors=True)
        if "images" in self.process_keys:
            for cam in self.cam_list:
                os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/image/{cam}"), exist_ok=True)
        if "calib" in self.process_keys:
            for cam in self.cam_list:
                os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/cam2world/{cam}"), exist_ok=True)
            os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/intrinsic"), exist_ok=True)
        if "lidar" in self.process_keys:
            os.makedirs(f"{self.save_dir}/{self.scene_name[i]}/lidar", exist_ok=True)
        if "dynamic_masks" in self.process_keys:
            for cam in self.cam_list:
                os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/dynamic_mask/{cam}"), exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser("Waymo to custom format (per-scene folders; world-coord cam2world & lidar)")
    ap.add_argument("--data_root", default="/media/fengyi/bb/waymo_raw", help="Waymo data root containing .tfrecord files")
    ap.add_argument("--target_dir", default="Data/waymo", help="Output root")


    return ap.parse_args()


def main():
    args = parse_args()
    dataset_processor = WaymoProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
    )
    for file_idx in range(len(dataset_processor)):
        dataset_processor.convert_one(file_idx)


if __name__ == "__main__":
    main()

