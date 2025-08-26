# Guidance Sequence Rendering

Here we give an example to show how we render guidance sequences. We assume that we have a 3D textured character, which can be reconstructed by [Wonder3D++](https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus). And we have used [Mixamo](https://www.mixamo.com/) to rig and retarget it to generate .fbx and .dae files.

## Install Blender
```sh
# download Blender for guidance sequence rendering
wget https://download.blender.org/release/Blender3.6/blender-3.6.14-linux-x64.tar.xz
tar -xvf blender-3.6.14-linux-x64.tar.xz
# install trimesh for blender's python
wget https://bootstrap.pypa.io/get-pip.py
./blender-3.6.14-linux-x64/3.6/python/bin/python3.10 get-pip.py
./blender-3.6.14-linux-x64/3.6/python/bin/python3.10 -m pip install trimesh
```

## Git Clone UniPose
```sh
git clone --recursive https://github.com/IDEA-Research/X-Pose.git unipose
```

## SDI Mask Coloring
```sh
python color.py
```

## Coarse Color and Mask Guidance Rendering
```sh
python animation_render.py
```

## Pose Guidance Rendering
```sh
python rotation_pose_rendering.py
python dae_pose_rendering.py
```
