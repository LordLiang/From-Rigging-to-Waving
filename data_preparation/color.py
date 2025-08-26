import trimesh
import cv2
import numpy as np
import argparse
import os


def nearest_color(image, xy):
    # 获取图像尺寸
    height, width = image.shape[0:2]

    # 将坐标拆分为整数和小数部分
    x_int = np.floor(xy[:,0]).astype(int)
    y_int = np.floor(xy[:,1]).astype(int)

    # 确保坐标不超出图像边界
    x_int = np.clip(x_int, 0, width - 2)
    y_int = np.clip(y_int, 0, height - 2)

    # 获取最近像素的颜色值
    value = image[y_int, x_int]
    return value


def get_mask_color(vertices, mask, res=512):
    mask = np.stack([mask] * 3, axis=-1)
    tmp = vertices[:, 0:2] / 1.35
    tmp[:,1] *= -1
    tmp = (tmp + 0.5) * (res-1)
    vert_weight_colors = nearest_color(mask, tmp) / 255.
    return vert_weight_colors


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
def delation_mask(inpaint_mask, mask):
    inpaint_mask_dilate = cv2.dilate(inpaint_mask, kernel, iterations=1)
    inpaint_mask = mask * inpaint_mask + (1-mask)*inpaint_mask_dilate
    return inpaint_mask.astype(np.uint8)

def color(trimesh_obj, data_dir):
    sdi_mask_front_fn = os.path.join(data_dir, 'char', 'sdi_mask_front.png')
    sdi_mask_back_fn = os.path.join(data_dir, 'char', 'sdi_mask_back.png')
    mask_fn = os.path.join(data_dir, 'char', 'mask.png')

    if os.path.exists(sdi_mask_front_fn):
        sdi_mask_front = cv2.imread(sdi_mask_front_fn, 0)
        _, sdi_mask_front = cv2.threshold(sdi_mask_front, 127, 255, cv2.THRESH_BINARY)
    else:
        print('Exit since no adi mask.')
        return
    
    if os.path.exists(sdi_mask_back_fn):
        sdi_mask_back = cv2.imread(sdi_mask_back_fn, 0)
        _, sdi_mask_back = cv2.threshold(sdi_mask_back, 127, 255, cv2.THRESH_BINARY)
    else:
        sdi_mask_back = sdi_mask_front

    mask = cv2.imread(mask_fn, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    sdi_mask_front = delation_mask(sdi_mask_front, mask)
    sdi_mask_back = delation_mask(sdi_mask_back, mask)
    vertices = trimesh_obj.vertices
    # M
    vert_weight_colors = np.ones(vertices.shape)
    vert_weight_colors[vertices[:,2]>0] = get_mask_color(vertices[vertices[:,2]>0], sdi_mask_front)
    vert_weight_colors[vertices[:,2]<=0] = get_mask_color(vertices[vertices[:,2]<=0], sdi_mask_back)
    # Masked color = C * (1 -  M)
    vert_colors = trimesh_obj.visual.vertex_colors
    vert_colors = vert_colors[:,0:3] / 255.
    vert_colors = vert_colors * (1 - vert_weight_colors)

    return vert_weight_colors, vert_colors



########################################

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="./data_example")
    ap.add_argument("--uid", type=str,  default="sketch-05")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--scale_factor", type=float, default=1.35)
    args = ap.parse_args()

    data_dir = os.path.join(args.dataset_root, args.uid)
    mesh_fn = os.path.join(data_dir, 'mesh', 'textured_mesh.obj')
    trimesh_obj = trimesh.load_mesh(mesh_fn, process=False)
    vert_weight_colors, vert_colors = color(trimesh_obj, data_dir)
    np.save(os.path.join(data_dir, 'mesh', 'textured_mesh_sdi_mask.npy'), vert_weight_colors)
    np.save(os.path.join(data_dir, 'mesh', 'textured_mesh_masked_color.npy'), vert_colors)
