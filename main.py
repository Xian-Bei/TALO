import os
import argparse
import random
import shutil

import ipdb
import numpy as np
import torch
from vggt_slam.solver import Solver
from VFMs_adaptor import initialize_model
from tqdm import trange

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--data_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--log_path", type=str, default="poses", help="Path to save the log file")
parser.add_argument("--port", type=int, default=8080, help="Port for the viewer")

parser.add_argument("--conf_threshold", type=int, default=60, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--interframe_solver_choice", default="tps", type=str, choices=['sim3', 'sl4', 'tps'])
parser.add_argument("--model", type=str, default="VGGT", choices=["VGGT", "Pi3", "MapAnything"], help="Choice of 3D foundation model")
parser.add_argument("--cam_num", type=int, default=6, help="Number of cameras used in the multi-camera rig")
parser.add_argument("--submap_size", type=int, default=2, help="Number of frames per submap")
parser.add_argument("--max_loops", type=int, default=0, help="Maximum number of loop closures per submap")


parser.add_argument("--vis_tps", action="store_true", help="Visualize the TPS allignment")
parser.add_argument("--allow_nonvalid", default=False, action="store_true", help="Allow adding submaps even if no valid points are found")
parser.add_argument("--backward_control_points", default=False, action="store_true", help="Use backward control points for SL4")
parser.add_argument("--filter_method", type=str, default="gaussian", choices=["mean", "gaussian", "robust", None, 'None'], help="Method for robust mean filtering of control points")
parser.add_argument("--filter_k", type=int, default=32, help="Number of nearest neighbors to use for robust mean filtering of control points")
parser.add_argument("--robust_mean_method", type=str, default='mad', choices=["mad", "std", None, 'None'], help="Method for robust mean filtering of control points")
parser.add_argument("--disable_sky_mask", action="store_true")


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    print("Using sky mask:", not args.disable_sky_mask)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if 'nuscenes' in args.data_folder.lower():
        cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]
    elif 'waymo' in args.data_folder.lower():
        cam_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    else:
        cam_names = sorted(os.listdir(os.path.join(args.data_folder, "image")))
    if args.cam_num < len(cam_names):
        cam_names = cam_names[:args.cam_num]
    cam_num = len(cam_names)
    submap_size = overlap_size = args.submap_size
    ref_cam_id = 0
    zero_pad = 3

    voxel_size_ratio = 0.05
    projection_error_ratio = 0.05
    max_loops = args.max_loops * cam_num  
    solver = Solver(
        model=args.model,
        init_conf_threshold=args.conf_threshold,
        visualize_global_map=args.vis_map,
        interframe_solver_choice=args.interframe_solver_choice,
        gradio_mode=False,
        port=args.port,
        save_root=args.log_path,
        overlap_size=overlap_size,
        submap_size=submap_size,
        cam_names=cam_names,
        ref_cam_id=ref_cam_id,
        ref_cam_name=cam_names[ref_cam_id],
        vis_tps=args.vis_tps,
        allow_nonvalid=args.allow_nonvalid,
        voxel_size_ratio=voxel_size_ratio,
        projection_error_ratio=projection_error_ratio,
        filter_method=args.filter_method,
        filter_k=args.filter_k,
        robust_mean_method=args.robust_mean_method,
        backward_control_points=args.backward_control_points,
    )
    
    print("Initializing and loading 3D foundation model...")
    model = initialize_model(args, device)

    timestep_num = len(os.listdir(os.path.join(args.data_folder, f"image/{cam_names[0]}")))
    frame_ids = []
    print(f"Found {timestep_num} timesteps")
    shutil.rmtree(args.log_path, ignore_errors=True)
    os.makedirs(args.log_path, exist_ok=True)

    for timestep in trange(timestep_num, desc="Timesteps"):
        frame_ids += [f"{cam_name}/{timestep:0>{zero_pad}d}" for cam_name in cam_names]  # assuming each camera has the same number of images
        if len(frame_ids) < submap_size * cam_num + overlap_size * cam_num and timestep < timestep_num - 1:
            continue
        image_names_subset = [f"{args.data_folder}/image/{frame_id}.jpg" for frame_id in frame_ids]  # assuming each camera has the same number of images

        predictions = solver.run_predictions(image_names_subset, model, max_loops=max_loops, frame_ids=frame_ids, sky_mask=not args.disable_sky_mask)

        solver.add_points(predictions)

        solver.graph.optimize()
        solver.map.update_submap_homographies(solver.graph)

        if args.vis_map:
            solver.cal_rig()
            solver.update_all_submap_vis()
            
        frame_ids = frame_ids[-overlap_size*cam_num:]

    if "tps" in args.interframe_solver_choice:
        solver.fit_tps_global()

    solver.cal_rig()
    
    if args.vis_map:
        solver.update_all_submap_vis(True)

    solver.write_to_each_camera()


if __name__ == "__main__":
    main()
