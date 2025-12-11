<div align="center">
    <a href="https://arxiv.org/abs/2512.02341"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
</div>

# TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction

<!-- # Updates

* **01.12.2025 — Initial Release** -->

> *We are still gradually removing redundant code and refining the implementation. More documentation and examples will be added soon.*

# Installation

## 1. Install TALO (Based on [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM))

```
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ unzip   # required by VGGT-SLAM
git clone https://github.com/TODO/talo
cd talo
conda create -n talo python=3.11
conda activate talo
./setup.sh   # installs third-party dependencies required by VGGT-SLAM 
```



## 2. Install Additional 3D Vision Foundation Models (Optional)

TALO currently supports the following 3D Vision Foundation Models as interchangeable backbones.

| Backbone                                                            | Installation                                                                                                                                                 |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[VGGT](https://github.com/facebookresearch/vggt)**                | Already installed through VGGT-SLAM                                                                                                                          |
| **[Pi3](https://github.com/yyfz/Pi3)**                              | Clone the repo into the TALO directory (e.g., `TALO/Pi3/pi3`)                                                                                                |
| **[MapAnything](https://github.com/facebookresearch/map-anything)** | Install `mapanything` as a package into the created `talo` conda environment  ([instructions](https://github.com/facebookresearch/map-anything?tab=readme-ov-file#installation)) |


It is also easy to integrate more advanced 3DVFMs by only formatting the prediction as a python dictionary containing the following keys
(see `VFMs_adaptor.py` for example implementations):
* `"org_images"`
* `"images"`
* `"cam2world"`
* `"intrinsic"`
* `"world_points"`
* `"world_points_conf"`


# Dataset Preparation

TALO currently supports:

* **nuScenes**
* **Waymo Open Dataset**


## 1. Download Raw Datasets

### Waymo Open Dataset

Download raw `.tfrecord` sequences:
[https://waymo.com/open/](https://waymo.com/open/)

### nuScenes

Download the full dataset:
[https://www.nuscenes.org/download](https://www.nuscenes.org/download)


## 2. Convert Raw Data into TALO Format

After downloading, convert datasets using our extraction scripts:

```
python extract_waymo.py
python extract_nuscenes.py
```

> Note that parsing Waymo requires `waymo-open-dataset-tf-2-6-0`, which depends on older package versions (e.g., `NumPy 1.x`) and is not compatible with the TALO environment.
Therefore, please create a separate Python environment specifically for extracting Waymo.

Please modify `data_root` and `save_root` accordingly in each script.

These scripts will:

* Extract RGB images (as model input)
* Extract camera intrinsics, extrinsics, and LiDAR (used as GT for evaluation)

### Output directory format

```
scene_dir/
  image/
    FRONT/
      000.jpg
      ...
  cam2world/
    FRONT/
      000.txt        # 4x4 matrix
      ...
    ...
  intrinsic/
    FRONT.txt        # 3x3 matrix
    ...
  lidar/
    000.bin
    ...
```

## Custom Data

To run the system on your own data, format it as follows:

```
custom_data
  example_scene/
    image/
      cam0/
        000.jpg
        ...
      ...
```

Then run:

```
python main.py --data_folder ./Data/custom_data/example_scene/ --log_path ./Save/custom_data/example_scene/VGGT+60+tps
```

for reconstruction and

```
python eval_vis_pcd_traj.py --GT ./Data/custom_data/example_scene/ --pred ./Save/custom_data/example_scene/VGGT+60+tps --vis
```
for visualization.
# Run and Evaluation

We provide a quick-start script that runs TALO on both Waymo and nuScenes, and summarizes results as reported in the paper.

```
bash run.sh
```


## `main.py` — Key Arguments

| Argument                     | Description                             |
| ---------------------------- | --------------------------------------- |
| `--data_folder`              | Path to the prepared scene directory    |
| `--log_path`                 | Directory to save logs/results          |
| `--model`                    | Choose from `{VGGT, Pi3, MapAnything}`  |
| `--conf_threshold`           | Confidence threshold for filtering      |
| `--interframe_solver_choice` | Choose from `{sim3, sl4, tps}`          |
| `--submap_size`              | Number of frames per submap             |
| `--cam_num`                  | Number of cameras to use                |
<!-- | `--vis_map`                  | Enable incremental online visualization | -->


# Visualization

TALO provides both online and offline visualization modes.


## 1. Incremental Online Visualization (viser)

Online **VGGT-SLAM** visualization by adding:

```
--vis_map
```

to `main.py` (inside `run.sh`).


## 2. Offline Full Visualization (Open3D)

Enable offline reconstruction visualization by adding:

```
--vis
```

to `eval_vis_pcd_traj.py` (inside `run.sh`).

## Examples
### VGGT on Waymo
![alt text](assets/vggt_61.png)
### VGGT on nuScenes
![alt text](assets/vggt_94.png)
### Pi3 on Waymo
![alt text](assets/pi3_31561.png)
### Pi3 on nuScenes
![alt text](assets/pi3_39.png)

# Acknowledgements
To ensure fair comparisons between different submap alignment methods (SL4 from [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) and Sim3 from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)), TALO is built upon the same framework (VGGT-SLAM) and extended to support multi-camera settings as well as additional 3D Vision Foundation Models (3DVFMs), including
[VGGT](https://github.com/facebookresearch/vggt),
[Pi3](https://github.com/yyfz/Pi3), and
[MapAnything](https://github.com/facebookresearch/map-anything).
All rights of these projects are fully reserved by their respective authors.

We sincerely thank the authors and maintainers of these outstanding open-source projects. If you find TALO useful, please consider citing and starring our work, and supporting the projects that made it possible.

