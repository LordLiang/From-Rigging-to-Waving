import bpy
import os
import sys
import argparse
import numpy as np
import trimesh
import json
from mathutils import Vector


def render_color_and_mask(anim_file, mesh_file, data_dir, output_dir):
    # load animation
    if anim_file.endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=anim_file)
    elif anim_file.endswith('.dae'):
        bpy.ops.wm.collada_import(filepath=anim_file)
    else:
        quit()

    armature = bpy.context.object
    armature.scale = (1, 1, 1)

    if anim_file.split('/')[-1] == 'rest_pose.fbx':
        mixamorig_kpts = {}
        for bone in armature.pose.bones:
            name = bone.name.replace(':', '_')
            mixamorig_kpts[name] = np.array(bone.head).tolist()
        with open(os.path.join(data_dir, 'mesh/mixamo_files/mixamorig_kpts.json'), 'w') as f:
            json.dump(mixamorig_kpts, f)
    else:
        print('you can rotate the character to change the viewpoint if you want')
        # armature.delta_rotation_euler[2] = np.radians(45)

    for selected_object in bpy.context.selected_objects:
        if selected_object.type == 'MESH':
            mesh_obj = selected_object

    mesh = mesh_obj.data
    faces = mesh.polygons
    indices = np.array([face.vertices for face in faces])

    trimesh_obj = trimesh.load_mesh(mesh_file)
    vert_colors = (trimesh_obj.visual.vertex_colors)[:,0:3] / 255.
    vert_colors_1 = np.load(mesh_file.replace('.obj', '_sdi_mask.npy'))
    vert_colors_2 = np.load(mesh_file.replace('.obj', '_masked_color.npy'))

    # repaint weight automatically
    armature.data.pose_position = 'REST'
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    bpy.ops.paint.weight_from_bones(type='AUTOMATIC')
    bpy.ops.object.mode_set(mode='OBJECT')
    armature.data.pose_position = 'POSE'

    animation_data = armature.animation_data
    if animation_data:
        # adjust the start frame
        old_start= int(animation_data.action.frame_range[0])
        if old_start != 1:
            for fcurve in animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.co.x += 1 - old_start

        bpy.context.scene.frame_start = int(animation_data.action.frame_range[0])
        bpy.context.scene.frame_end = int(animation_data.action.frame_range[1])

        # adjust the view window
        delta_location_file = anim_file.replace('.fbx', '.txt')
        if os.path.exists(delta_location_file):
            armature.delta_location = np.loadtxt(delta_location_file)
        else:
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
                bpy.context.scene.frame_set(frame)
                bbox = bpy.context.selected_objects[1].bound_box
                matrix_world =  bpy.context.selected_objects[1].matrix_world
                for point in bbox:
                    world_point = matrix_world @ Vector(point)
                    min_x = min(min_x, world_point[0])
                    max_x = max(max_x, world_point[0])
                    min_y = min(min_y, world_point[1])
                    max_y = max(max_y, world_point[1])
                    min_z = min(min_z, world_point[2])
                    max_z = max(max_z, world_point[2])

            # translate mesh to the center position
            armature.delta_location = [-(max_x + min_x)/2, max_y - min_y, -(max_z + min_z)/2]
            np.savetxt(anim_file.replace('.fbx', '.txt'), armature.delta_location, fmt="%.4f")
    
    # 遍历并删除所有顶点颜色层
    while mesh.vertex_colors:
        mesh.vertex_colors.remove(mesh.vertex_colors[0])

    material = bpy.data.materials.new(name='VertexColorMaterial')
    mesh.materials.append(material)
    mesh.vertex_colors.new(name='VertexColors')
    vertex_colors = mesh.vertex_colors["VertexColors"]
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    output_node = nodes.get('Material Output')
    links.new(vertex_color_node.outputs[0], output_node.inputs[0])

    bpy.context.scene.view_settings.view_transform = 'Standard'

    # color
    colors = vert_colors[indices].reshape(-1, 3)
    save_path = os.path.join(output_dir, 'color')
    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    for i, color in enumerate(colors):
        vertex_colors.data[i].color = (color[0], color[1], color[2], 1)
    bpy.ops.render.render(animation=True)
    
    # sdi mask
    colors = vert_colors_1[indices].reshape(-1, 3)
    save_path = os.path.join(output_dir, 'mask')
    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    for i, color in enumerate(colors):
        vertex_colors.data[i].color = (color[0], color[1], color[2], 1)
    bpy.ops.render.render(animation=True)

    # masked
    colors = vert_colors_2[indices].reshape(-1, 3)
    save_path = os.path.join(output_dir, 'masked')
    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    for i, color in enumerate(colors):
        vertex_colors.data[i].color = (color[0], color[1], color[2], 1)
    bpy.ops.render.render(animation=True)

if __name__ == '__main__':
    try:
        idx = sys.argv.index("--")
        script_args = sys.argv[idx + 1:]
    except ValueError as e:  # '--' not in the list:
        script_args = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--anim_file', type=str, help='path to animation file', required=True)
    parser.add_argument('--mesh_file', type=str, help='path to mesh file', required=True)
    parser.add_argument('--input_dir', type=str, help='path to load obj', required=True)
    parser.add_argument('--output_dir', type=str, help='path to save renderings', required=True)
    
    args = parser.parse_args(script_args)
    render_color_and_mask(args.anim_file, args.mesh_file, args.input_dir, args.output_dir)
