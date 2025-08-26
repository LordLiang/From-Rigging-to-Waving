import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image
import argparse
import json
from draw_openpose_depth import draw_bodypose, openpose_keypoints, dae_to_openpose_map
from scipy.spatial.transform import Rotation as R


def rotate_relative_to_head(offset, rotation_matrix):
    return rotation_matrix @ offset


# Big boy function for stepping through the DAE file and applying animation transforms
def parse_dae_for_visual_scene(dae_file_path, scale_factor=1.35, res=512):
    tree = ET.parse(dae_file_path)
    root = tree.getroot()
    namespace = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    visual_scene = root.find('.//collada:library_visual_scenes/collada:visual_scene', namespace)
    animations = root.find('.//collada:library_animations', namespace)
    if visual_scene is None or animations is None:
        raise ValueError("Visual scene or animation data not found in DAE file.")

    # Dictionary to store animation data
    animation_data = {}
    
    # Parse the animation transforms
    for animation in animations.findall('collada:animation', namespace):
        target_id = animation.get('id').split('-')[0]
        times = animation.find(f'.//collada:source[@id="{target_id}-Matrix-animation-input"]/collada:float_array', namespace)
        transforms = animation.find(f'.//collada:source[@id="{target_id}-Matrix-animation-output-transform"]/collada:float_array', namespace)
        
        if times is None or transforms is None:
            continue

        time_values = list(map(float, times.text.split()))
        transform_values = list(map(float, transforms.text.split()))
        matrices = [np.array(transform_values[i:i+16]).reshape(4, 4) for i in range(0, len(transform_values), 16)]
        
        animation_data[target_id] = (time_values, matrices)

    frames = {i: {} for i in range(len(next(iter(animation_data.values()))[0]))}

    # Recursively parse nodes in the visual scene and apply animations
    def parse_joint(node, parent_transform, time_idx):
        joint_name = node.get('id')
        openpose_name = dae_to_openpose_map.get(joint_name)

        # Use animation transform if available, else fall back to the node's local transform
        if joint_name in animation_data:
            _, matrices = animation_data[joint_name]
            local_transform = matrices[time_idx]
        else:
            matrix_text = node.find('collada:matrix', namespace).text
            matrix_values = list(map(float, matrix_text.split()))
            local_transform = np.array(matrix_values).reshape(4, 4)
        
        # Compute the world transform for the joint at this time step
        world_transform = np.dot(parent_transform, local_transform)
        x, y, z = world_transform[0, 3], -world_transform[1, 3], world_transform[2, 3]

        if openpose_name:
            frames[time_idx][openpose_name] = [x, y, z]
            if openpose_name == "nose":  # The head's orientation is based on the nose position
                frames[time_idx]["head_rotation_matrix"] = world_transform[:3, :3]

        for child in node.findall('collada:node', namespace):
            parse_joint(child, world_transform, time_idx)

    # Parse the visual scene for each time step
    for time_idx in range(len(next(iter(animation_data.values()))[0])):
        root_node = visual_scene.find('.//collada:node[@id="mixamorig_Hips"]', namespace)
        parse_joint(root_node, np.eye(4), time_idx)

    # read facial kpts
    with open(os.path.join(os.path.dirname(dae_file_path), 'mixamorig_kpts.json'), 'r') as f:
        mixamorig_kpts = json.load(f)

    top_initial = np.array(mixamorig_kpts["mixamorig_HeadTop_End"]) * np.array([1,-1,1])
    head_initial = np.array(mixamorig_kpts["mixamorig_Head"]) * np.array([1,-1,1])
    middle_initial = (top_initial + head_initial) / 2

    nose_initial = np.array(mixamorig_kpts["mixamorig_Nose"]) * np.array([1,-1,1])
    right_eye_initial = np.array(mixamorig_kpts["mixamorig_RightEye"]) * np.array([1,-1,1])
    left_eye_initial = np.array(mixamorig_kpts["mixamorig_LeftEye"]) * np.array([1,-1,1])
    right_ear_initial = np.array(mixamorig_kpts["mixamorig_RightEar"]) * np.array([1,-1,1])
    left_ear_initial = np.array(mixamorig_kpts["mixamorig_LeftEar"]) * np.array([1,-1,1])
    nose_offset_scaled = nose_initial - middle_initial
    right_eye_offset_scaled = right_eye_initial - middle_initial
    left_eye_offset_scaled = left_eye_initial - middle_initial
    right_ear_offset_scaled = right_ear_initial - middle_initial
    left_ear_offset_scaled = left_ear_initial - middle_initial

    # Add facial feature points based on the head orientation and position
    for time_idx, frame in frames.items():
        top = np.array(frame.get("nose"))
        head = np.array(frame.get("head"))
        middle = (top + head) / 2
        head_rotation_matrix = frame.get("head_rotation_matrix")

        # Convert rotation matrix to Euler angles and adjust yaw if needed
        rotation = R.from_matrix(head_rotation_matrix)
        yaw, pitch, roll = rotation.as_euler('zyx', degrees=True)
        yaw *= -1  # Invert yaw
        roll *= -1  # Invert roll
        adjusted_rotation_matrix = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

        rotated_nose = rotate_relative_to_head(nose_offset_scaled, adjusted_rotation_matrix) + middle
        rotated_right_eye = rotate_relative_to_head(right_eye_offset_scaled, adjusted_rotation_matrix) + middle
        rotated_left_eye = rotate_relative_to_head(left_eye_offset_scaled, adjusted_rotation_matrix) + middle
        rotated_right_ear = rotate_relative_to_head(right_ear_offset_scaled, adjusted_rotation_matrix) + middle
        rotated_left_ear = rotate_relative_to_head(left_ear_offset_scaled, adjusted_rotation_matrix) + middle

        frame["nose"] = rotated_nose.tolist()
        frame["right_eye"] = rotated_right_eye.tolist()
        frame["left_eye"] = rotated_left_eye.tolist()
        frame["right_ear"] = rotated_right_ear.tolist()
        frame["left_ear"] = rotated_left_ear.tolist()

    # global tranformation
    delta_location = np.loadtxt(dae_file_path.replace('.dae', '.txt'))
    # Rotate and scale all points in the frames
    frames = rotate_and_scale_pose(frames, delta_location, scale_factor, res)
    return frames


# Apply rotations to all keypoints around the image center
def rotate_and_scale_pose(frames, delta_location, scale_factor=1.35, res=512, rotation_angles=(0, 0, 0)):
    # Convert angles to radians
    rx, ry, rz = np.radians(rotation_angles)
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix in the order: Ry -> Rx -> Rz
    R = Rz @ (Rx @ Ry)

    # Apply rotation and scaling to each point in each frame
    for time_idx, frame in frames.items():
        for point_key, coords in frame.items():
            # Ensure coords is a list of three elements before proceeding
            if isinstance(coords, list) and len(coords) == 3:
                x, y, z = coords
                
                # Apply scaling
                x = (x + delta_location[0]) * res / scale_factor + res // 2
                y = (y - delta_location[2]) * res / scale_factor + res // 2
                z = z * res / scale_factor
                
                # Rotate the point around the origin (0, 0, 0)
                rotated_point = R @ np.array([x, y, z])
                
                # Update the point with rotated and scaled coordinates
                frame[point_key] = rotated_point.tolist()

    return frames


# Format frame keypoints to OpenPose JSON standard
def format_to_openpose(frames):
    formatted_frames = []
    for time_idx, frame in frames.items():
        candidate = []
        
        for keypoint in openpose_keypoints:
            if keypoint in frame:
                x, y, z = frame[keypoint]
                candidate.append([x, y, z])
            else:
                candidate.append([0.0, 0.0, 0.0])

        formatted_frames.append(np.array(candidate))
    
    return formatted_frames


# Convert DAE to OpenPose frames with rotation and scaling
def convert_dae_to_openpose(dae_file, scale_factor=2, res=512):
    frames = parse_dae_for_visual_scene(dae_file, scale_factor=scale_factor, res=res)
    openpose_frames = format_to_openpose(frames)
    return openpose_frames


#Convert dae file and save as image sequence
def convert_dae(data_dir, action_name, scale_factor, res):
    
    dae_file = os.path.join(data_dir, 'mesh/mixamo_files', action_name+'.dae')
    frames = convert_dae_to_openpose(dae_file, scale_factor, res)

    image_folder = os.path.join(data_dir, 'blender_render', action_name, 'color')
    openpose_folder = os.path.join(data_dir, 'blender_render', action_name, 'openpose')
    openpose_color_folder = os.path.join(data_dir, 'blender_render', action_name, 'openpose_color')
    os.makedirs(openpose_folder, exist_ok=True)
    os.makedirs(openpose_color_folder, exist_ok=True)

    for idx, kpts in enumerate(frames):
        img_path = os.path.join(image_folder, f"{idx+1:04}.png")
        img = Image.open(img_path)

        # openpose
        canvas = np.zeros(shape=(res, res, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, kpts)
        canvas.save(os.path.join(openpose_folder, f"{idx+1:04}.png"))
        img2 = canvas.convert('RGBA')
        img2.putalpha(128)
        combined = Image.alpha_composite(img, img2)
        combined.save(os.path.join(openpose_color_folder, f"{idx+1:04}.png"))

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="./data_example")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--scale_factor", type=float, default=1.35)
    ap.add_argument("--uid", type=str, default="sketch-05")
    args = ap.parse_args()

    data_dir = os.path.join(args.dataset_root, args.uid)

    action_types = []
    for item in os.listdir(os.path.join(data_dir, 'mesh', 'mixamo_files')):
        if item.endswith('.dae'):
            action_types.append(item.replace('.dae', ''))

    for action_type in action_types:
        convert_dae(data_dir, action_type, args.scale_factor, args.res)
