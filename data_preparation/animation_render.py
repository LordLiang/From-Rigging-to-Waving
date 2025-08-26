import subprocess
import os
import argparse
import time
import glob


if __name__ == '__main__':    

    # run 'export DISPLAY=:1' if terminal rendering
    parser = argparse.ArgumentParser(description='guidance sequence rendering')
    parser.add_argument("--dataset_root", type=str, default="./data_example")
    parser.add_argument('--uid', type=str, default='sketch-05', help='image uid')
    parser.add_argument('--blender_install_path', type=str, default='./blender-3.6.14-linux-x64/blender', help='blender path')
    parser.add_argument('--engine_type', type=str, default='BLENDER_EEVEE', help='BLENDER_EEVEE/CYCLES')
    args = parser.parse_args()
    
    input_dir = os.path.join(args.dataset_root, args.uid)
    script_file = './blender_related/script.py'
    mixamo_dir = os.path.join(input_dir, 'mesh/mixamo_files')
    mesh_file = os.path.join(input_dir, 'mesh/textured_mesh.obj')

    action_types = ['rotation']
    for item in os.listdir(mixamo_dir):
        if item.endswith('.fbx') and item != 'rest_pose.fbx':
            action_types.append(item.replace('.fbx', ''))
    
    for action_type in action_types:
        output_dir = os.path.join(input_dir, 'blender_render', action_type)
        start = time.time()

        if not os.path.exists(os.path.join(output_dir, 'color')):
            if action_type == 'rotation':
                anim_file = os.path.join(mixamo_dir, 'rest_pose.fbx')
                config_file = './blender_related/config_ortho_rotate.blend'
            else:
                anim_file = os.path.join(mixamo_dir, '%s.fbx'%(action_type))
                config_file = './blender_related/config_ortho.blend'

            subprocess.run(f'{args.blender_install_path} -b {config_file} -E {args.engine_type} --python {script_file} \
                                                        -- --anim_file {anim_file} \
                                                           --mesh_file {mesh_file} \
                                                           --output_dir {output_dir} \
                                                           --input_dir {input_dir}', shell=True)

        end = time.time()
        num_frame = len(glob.glob(os.path.join(output_dir, 'color', '*.png')))
        print((end-start)/num_frame, num_frame)
        
