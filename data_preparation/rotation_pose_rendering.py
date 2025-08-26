import os
import numpy as np
import argparse
import trimesh
import mesh_raycast
from PIL import Image
import math
import json

from draw_openpose_depth import draw_bodypose, openpose_to_dae_map, openpose_keypoints
from unipose_utils import load_model, run_unipose


def adjust_kpts(kpts, mask):
    res = mask.shape[0]
    kpts = (kpts*res).astype(int)
    kpts_new = []
    for i in range(5):
        point = kpts[i]
        # print(point, '1')
        # Check if the point is inside the mask
        if mask[point[1], point[0]] == 0:
            # print(point, '2')
            # Find the closest point inside the mask
            mask_indices = np.argwhere(mask)
            mask_indices = mask_indices[:,[1,0]]
            distances = [math.sqrt((p[1] - point[1])**2 + (p[0] - point[0])**2) for p in mask_indices]
            closest_point_index = np.argmin(distances)
            closest_point = mask_indices[closest_point_index]
            point = closest_point
        kpts_new.append(point)
    kpts_new = np.array(kpts_new, dtype=np.float32)
    kpts_new = kpts_new / (res-1) - 0.5
    kpts_new[:, 1] *= -1
    return kpts_new


def get_kpts_3d(kpts, triangles, mask):
    kpts = adjust_kpts(kpts, mask)
    kpts_3d = np.zeros((5, 3))
    kpts_3d[:, 0:2] = kpts[:, 0:2]
    for i in range(5):
        source = (kpts[i, 0], kpts[i, 1], 1)
        rs = mesh_raycast.raycast(source, (0,0,-1), mesh=triangles)
        if len(rs) > 0:
            kpts_3d[i, 2] = min(rs, key=lambda x: x['distance'])['point'][2]
    return kpts_3d


def rotate_point_cloud_Y(point_cloud, theta):
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotated_cloud = np.dot(point_cloud, R_y.T)
    return rotated_cloud


def step(data_dir, scale_factor, res):
    mesh_file = os.path.join(data_dir, 'mesh', 'textured_mesh.obj')
    mesh = trimesh.load(mesh_file)
    vertices, faces = np.array(mesh.vertices) / scale_factor, np.array(mesh.faces)
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')
    mask = np.array(Image.open(os.path.join(data_dir, 'blender_render/rotation/color/0001.png')))[:,:,3]
    unipose_model = load_model(model_config_path='./unipose/config_model/UniPose_SwinT.py',
                               model_checkpoint_path='./unipose/weights/unipose_swint.pth')
    
    img_path = os.path.join(data_dir, 'char/texture.png')
    facial_kpts_2d = run_unipose(unipose_model, img_path)[:,0:5]
    facial_kpts_3d = get_kpts_3d(facial_kpts_2d, triangles, mask) * 1.35

    mixamorig_kpts_file = os.path.join(data_dir, 'mesh/mixamo_files/mixamorig_kpts.json')
    with open(mixamorig_kpts_file) as f:
        mixamorig_kpts = json.load(f)

    mixamorig_kpts['mixamorig_Nose'] = facial_kpts_3d[0].tolist()
    mixamorig_kpts['mixamorig_LeftEye'] = facial_kpts_3d[1].tolist()
    mixamorig_kpts['mixamorig_RightEye'] = facial_kpts_3d[2].tolist()
    mixamorig_kpts['mixamorig_LeftEar'] = facial_kpts_3d[3].tolist()
    mixamorig_kpts['mixamorig_RightEar'] = facial_kpts_3d[4].tolist()

    mixamorig_kpts_file = os.path.join(data_dir, 'mesh/mixamo_files/mixamorig_kpts.json')
    with open(mixamorig_kpts_file, 'w') as f:
        json.dump(mixamorig_kpts, f)

    kpts_3d = np.zeros((18, 3))

    for i in range(18):
        mixamorig_name = openpose_to_dae_map[openpose_keypoints[i]]
        kpts_3d[i] = np.array(mixamorig_kpts[mixamorig_name])

    kpts_3d /= scale_factor

    # render openpose for rotation frames
    image_folder = os.path.join(data_dir, 'blender_render/rotation/color')
    openpose_folder = os.path.join(data_dir, 'blender_render/rotation/openpose')
    openpose_color_folder = os.path.join(data_dir, 'blender_render/rotation/openpose_color')
    os.makedirs(openpose_folder, exist_ok=True)
    os.makedirs(openpose_color_folder, exist_ok=True)
    rotation_angles = np.loadtxt('./blender_related/rotation_angles.txt')

    for idx in range(60):
        img_path = os.path.join(image_folder, f"{idx+1:04}.png")
        img = Image.open(img_path)

        angle = - rotation_angles[idx]
        tmp = rotate_point_cloud_Y(kpts_3d, angle)
        tmp[:, 1] *= -1
        tmp = (tmp + 0.5) * (res-1)

        # openpose
        canvas = np.zeros(shape=(res, res, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, tmp)
        canvas.save(os.path.join(openpose_folder, f"{idx+1:04}.png"))
        img2 = canvas.convert('RGBA')
        img2.putalpha(128)
        combined = Image.alpha_composite(img, img2)
        combined.save(os.path.join(openpose_color_folder, f"{idx+1:04}.png"))

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="./data_example")
    ap.add_argument("--uid", type=str,  default="sketch-05")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--scale_factor", type=float, default=1.35)
    args = ap.parse_args()

    data_dir = os.path.join(args.dataset_root, args.uid)
    step(data_dir, args.scale_factor, args.res)
        
