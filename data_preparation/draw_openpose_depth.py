import numpy as np
import cv2
import math
from PIL import Image

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


openpose_keypoints = ["nose", "neck", "right_shoulder", "right_elbow", \
                      "right_wrist", "left_shoulder", "left_elbow", "left_wrist", \
                      "right_hip", "right_knee", "right_ankle", "left_hip", \
                      "left_knee", "left_ankle", "right_eye", "left_eye", 
                     "right_ear", "left_ear"]

mmpose_keypoints = ['nose', 'left eye', 'right eye', 'left ear', \
                    'right ear', 'left shoulder', 'right shoulder', 'left elbow', \
                    'right elbow', 'left wrist', 'right wrist', 'left hip', \
                    'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

def openpose_to_mmpose(openpose_kps):
    """
    将OpenPose关键点格式转换为MMPose关键点格式
    
    参数:
        openpose_kps: OpenPose格式的关键点列表或数组，顺序为OpenPose标准18个点
                     形状应为(18, 2)或(18, 3)（如果有置信度）
    
    返回:
        mmpose_kps: MMPose格式的关键点数组，形状为(17, 2)或(17, 3)
    """

    # 确保输入是numpy数组
    openpose_kps = np.array(openpose_kps)
    
    # OpenPose到MMPose的索引映射
    # OpenPose顺序: 0-nose, 1-neck, 2-Rshoulder, 3-Relbow, 4-Rwrist, 5-Lshoulder, 6-Lelbow, 7-Lwrist,
    #               8-Rhip, 9-Rknee, 10-Rankle, 11-Lhip, 12-Lknee, 13-Lankle, 14-Reye, 15-Leye, 16-Rear, 17-Lear
    # MMPose顺序:   0-nose, 1-Leye, 2-Reye, 3-Lear, 4-Rear, 5-Lshoulder, 6-Rshoulder, 7-Lelbow, 8-Relbow,
    #               9-Lwrist, 10-Rwrist, 11-Lhip, 12-Rhip, 13-Lknee, 14-Rknee, 15-Lankle, 16-Rankle
    
    # 创建映射关系
    openpose_indices = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    
    # 根据输入维度处理
    if openpose_kps.shape[-1] == 2:  # 只有(x,y)
        mmpose_kps = openpose_kps[openpose_indices]
    elif openpose_kps.shape[-1] == 3:  # (x,y,score)
        mmpose_kps = openpose_kps[openpose_indices, :]
    else:
        raise ValueError("输入关键点格式不正确，应为(18,2)或(18,3)")
    
    return mmpose_kps


# Mixamo DAE to OpenPose mappings
openpose_to_dae_map = {
    "headtop_end": "mixamorig_HeadTop_End",
    "head": "mixamorig_Head",
    "nose": "mixamorig_Nose", 
    "neck": "mixamorig_Neck", 
    "right_shoulder": "mixamorig_RightArm", 
    "right_elbow": "mixamorig_RightForeArm", 
    "right_wrist": "mixamorig_RightHand",
    "left_shoulder": "mixamorig_LeftArm", 
    "left_elbow": "mixamorig_LeftForeArm", 
    "left_wrist": "mixamorig_LeftHand", 
    "right_hip": "mixamorig_RightUpLeg", 
    "right_knee": "mixamorig_RightLeg",
    "right_ankle": "mixamorig_RightFoot", 
    "left_hip": "mixamorig_LeftUpLeg", 
    "left_knee": "mixamorig_LeftLeg", 
    "left_ankle": "mixamorig_LeftFoot", 
    "right_eye": "mixamorig_RightEye",
    "left_eye": "mixamorig_LeftEye", 
    "right_ear": "mixamorig_RightEar", 
    "left_ear": "mixamorig_LeftEar"
}

# Mixamo DAE to OpenPose mappings
dae_to_openpose_map = {
    "mixamorig_HeadTop_End": "nose",
    "mixamorig_Head": "head",  # Not used by OpenPose, exclusively for storing head position and angle for generating nose/eye/ear points
    "mixamorig_Neck": "neck",
    "mixamorig_RightArm": "right_shoulder",
    "mixamorig_RightForeArm": "right_elbow",
    "mixamorig_RightHand": "right_wrist",
    "mixamorig_LeftArm": "left_shoulder",
    "mixamorig_LeftForeArm": "left_elbow",
    "mixamorig_LeftHand": "left_wrist",
    "mixamorig_RightUpLeg": "right_hip",
    "mixamorig_RightLeg": "right_knee",
    "mixamorig_RightFoot": "right_ankle",
    "mixamorig_LeftUpLeg": "left_hip",
    "mixamorig_LeftLeg": "left_knee",
    "mixamorig_LeftFoot": "left_ankle",
    "mixamorig_RightEye": "right_eye",
    "mixamorig_LeftEye": "left_eye",
    "mixamorig_RightEar": "right_ear",
    "mixamorig_LeftEar": "left_ear"
}

# for openpose
def draw_bodypose(canvas, kpts, stickwidth=4):
    z = kpts[:,2]
    # 按深度（Z坐标）对骨架进行排序
    # 将 limbSeq 和 colors 组合成一个列表，方便一起排序
    limbSeq_colors = list(zip(limbSeq, colors))
    sorted_limbSeq_colors = sorted(limbSeq_colors, key=lambda item: (z[item[0][0]-1] + z[item[0][1]-1]) / 2)

    for i in range(17):
        index = np.array(sorted_limbSeq_colors[i][0]) - 1
        Y = kpts[index.astype(int), 0]
        X = kpts[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
 
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, sorted_limbSeq_colors[i][1])

    canvas = (canvas * 0.6).astype(np.uint8)

    # 按深度（Z坐标）对关键点进行排序
    index = np.arange(18)
    sorted_index = sorted(index, key=lambda item: z[item])

    for i in sorted_index:
        x, y = kpts[i][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return Image.fromarray(canvas)

############################# for comparing with MikuDance

limbSeq2 = [[16,14],[14,12],[17,15],[15,13],\
            [12,13],[6,12],[7,13],[6,7],\
            [6,8],[7,9],[8,10],[9,11],[2,3],\
            [1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
kpt_color = [[255, 255, 100], [255, 255, 100], [255, 255, 100], [255, 255, 100],
             [255, 255, 100], [255, 0, 0], [160, 32, 240], [255, 0, 0],
             [160, 32, 240], [255, 0, 0], [160, 32, 240], [0, 255, 0], [51, 153, 255],
             [0, 255, 0], [51, 153, 255], [0, 255, 0], [51, 153, 255]]
link_color = [[0, 255, 0], [0, 255, 0], [51, 153, 255], [51, 153, 255],
              [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 0, 0], [160, 32, 240], [255, 0, 0], [160, 32, 240],
              [255, 255, 100], [255, 255, 100], [255, 255, 100], [255, 255, 100],
              [255, 255, 100], [255, 255, 100], [255, 255, 100]]


def mmpose_to_openpose(kps):
    neck = np.mean(kps[[5, 6]], axis=1)
    new_kps = np.insert(kps, 17, neck, axis=1)

    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_kps[openpose_idx] = new_kps[mmpose_idx]
    return new_kps


# for mmpose
def draw_bodypose2(canvas, kpts, linewidth=4):
    kpts = openpose_to_mmpose(kpts)
    z = kpts[:,2]
    # 按深度（Z坐标）对骨架进行排序
    # 将 index 和 limbSeq 组合成一个列表，方便一起排序
    index_limbSeq = list(zip(np.arange(19), limbSeq2))
    sorted_index_limbSeq = sorted(index_limbSeq, key=lambda item: (z[item[1][0]-1] + z[item[1][1]-1]) / 2)

    for i, limb in sorted_index_limbSeq:
        p1 = kpts[limb[0]-1, 0:2]
        p2 = kpts[limb[1]-1, 0:2]
        points = np.array([[p1, p2]],dtype=np.int32)
        cv2.polylines(canvas, points, False, link_color[i], linewidth, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)
