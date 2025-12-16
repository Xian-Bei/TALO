from third_party_codes.vggt_code import *


VFM_REGESTRATION = {
    "VGGT": "NOSCALE",
    "Pi3": "NOSCALE",
    "MapAnything": "MECTRICSCALE"
}

def initialize_model(args, device):
    if args.model == "VGGT":
        from vggt.models.vggt import VGGT
        # model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model = model.to(device).eval()
        
    elif args.model == "Pi3":
        from Pi3.pi3.models.pi3 import Pi3 as Pi3Model
        model = Pi3Model.from_pretrained("yyfz233/Pi3").to(device).eval()
    elif args.model == "MapAnything":
        from mapanything.models import MapAnything
        model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
    return model


def run_predictions(image_names, model_name, model, sky_mask=False):
    if model_name == "VGGT":
        results = run_VGGT(image_names, model, sky_mask)
    elif model_name == "Pi3":
        results = run_Pi3(image_names, model, sky_mask)
    elif model_name == "MapAnything":
        results = run_MapAnything(image_names, model, sky_mask)
    return results

def run_VGGT(image_names, model, sky_mask=False):
    results = {}
    from vggt.utils.load_fn import load_and_preprocess_images as load_images
    from vggt.utils.geometry import closed_form_inverse_se3
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_images(image_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = images.to(device)
    results['org_images'] = images
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    results["images"] = (images.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    results["cam2world"] = closed_form_inverse_se3(extrinsic.cpu().numpy().squeeze(0))
    results["intrinsic"] = intrinsic.cpu().numpy().squeeze(0)
    results['world_points'] = predictions["world_points"].cpu().numpy().squeeze(0)  # (S, H, W, 3)
    results["world_points_conf"] = predictions["world_points_conf"].cpu().numpy().squeeze(0)
    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])  # (n, H, W)
        results["world_points_conf"] = (non_sky_mask_binary * results["world_points_conf"])

    return results



def run_Pi3(image_names, model, sky_mask=False):
    results = {}
    from Pi3.pi3.utils.geometry import homogenize_points
    from third_party_codes.pi3_code import load_images_as_tensor as load_images, recover_focal_no_shift
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_images(image_names)
    images = images.to(device)
    results['org_images'] = images
    images = images[None]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    cam2world = torch.inverse(predictions['camera_poses'][0][0]) @ predictions['camera_poses'][0] # (S, 4, 4)
    results['cam2world'] = cam2world.cpu().numpy() # (S, 4, 4)
    results['world_points'] = torch.einsum('nij, nhwj -> nhwi', cam2world, homogenize_points(predictions['local_points'][0]))[..., :3].cpu().numpy() # (S, H, W, 3)
    results['world_points_conf'] = torch.sigmoid(predictions['conf'][0,...,0]).cpu().numpy() # (S, H, W)
    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])  # (n, H, W)
        results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]
    results["images"] = (images[0].permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)


    points = predictions["local_points"] # (1, S, H, W, 3)
    masks = None
    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height
    # use recover_focal_shift function from MoGe
    focal = recover_focal_no_shift(points, masks) # focal: (1, S), shift: (1, S)
    fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    zero_point_five = torch.full_like(fx, 0.5)
    intrinsics = torch.stack([
        fx, zeros, zero_point_five, 
        zeros, fy, zero_point_five, 
        zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # (S, 3, 3)
    
    
    H, W = original_height, original_width

    intrinsics[:, 0, 0] *= (W - 1)   # fx
    intrinsics[:, 1, 1] *= (H - 1)   # fy
    intrinsics[:, 0, 2] *= (W - 1)   # cx
    intrinsics[:, 1, 2] *= (H - 1)   # cy
    results['intrinsic'] = intrinsics.cpu().numpy()

    return results




def run_MapAnything(image_names, model, sky_mask=False):
    from mapanything.utils.image import load_images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_images(image_names)
    predictions = model.infer(
        images,                            # Input views
        memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
        use_amp=True,                     # Use mixed precision inference (recommended)
        amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
        apply_mask=True,                  # Apply masking to dense geometry outputs
        mask_edges=True,                  # Remove edge artifacts by using normals and depth
        apply_confidence_mask=False,      # Filter low-confidence regions
        confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
    )
    results = {
        "cam2world": torch.cat([pred["camera_poses"] for pred in predictions], 0).cpu().numpy(),         # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
        "intrinsic": torch.cat([pred["intrinsics"] for pred in predictions], 0).cpu().numpy(),           # Recovered pinhole camera intrinsics (B, 3, 3)
        "world_points": torch.cat([pred["pts3d"] for pred in predictions], 0).cpu().numpy(),                     # 3D points in world coordinates (B, H, W, 3)
        "world_points_conf": torch.cat([(pred["conf"] * pred["mask"].squeeze(-1)) for pred in predictions], 0).cpu().numpy(),                   # Per-pixel confidence scores (B, H, W)
        "org_images": torch.cat([image['img'] for image in images]).to(device),
        "images": (torch.cat([pred["img_no_norm"] for pred in predictions], 0).cpu().numpy()*255).astype(np.uint8)                  # Denormalized input images for visualization (B, H, W, 3)
    }
    results['cam2world'] = np.linalg.inv(results['cam2world'][0]) @ results['cam2world'] # (S, 4, 4)
    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images[0]['true_shape'][0].tolist())  # (n, H, W)
        results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]

    return results