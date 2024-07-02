from pathlib import Path
import numpy as np
import sys; sys.path
import trimesh
import open3d as o3d
from typing import *
import h5py

def to_pcd(x, normals=None, color=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
    if color is not None:
        if color == 'red':
            color = [1, 0, 0]
        elif color == 'blue':
            color = [0, 0, 1]
        elif color == 'green':
            color = [0, 1, 0]
        pcd.paint_uniform_color(color)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def viz(x, frame=True, bb=0.25, bb_offset=None):
    if not isinstance(x, list):
        x = [x]
    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        x += [frame]
    if bb_offset is None:
        bb_offset = np.zeros(3)
    if bb is not None:
        x += [get_bounding_box(bb).translate(bb_offset)]
    o3d.visualization.draw_geometries(x,
                                      lookat=[0, 0, 0],
                                      up=[0, 0, 1],
                                      front=[1, 1, 1],
                                      zoom=1, )


def get_bounding_box(box_size):
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    points = (np.array(points) - 0.5) * box_size
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_gripper_marker(sections=6, scale=1.) -> trimesh.Trimesh:
    cfl = trimesh.creation.cylinder(
        radius=0.002 * scale,
        sections=sections,
        segment=[
            [0.041 * scale, 0, 0.066 * scale],
            [0.041 * scale, 0, 0.112 * scale],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002 * scale,
        sections=sections,
        segment=[
            [-0.041 * scale, 0, 0.066 * scale],
            [-0.041 * scale, 0, 0.112 * scale],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002 * scale,
        sections=sections,
        segment=[[0, 0, 0], [0, 0, 0.066 * scale]],
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002 * scale,
        sections=sections,
        segment=[[-0.041 * scale, 0, 0.066 * scale], [0.041 * scale, 0, 0.066 * scale]],
    )
    T_offset = np.array([
        [0., 1., 0., 0.],
        [-1., 0., -0., 0.],
        [-0., 0., 1., 0.105],
        [0., 0., 0., 1.]])

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp = tmp.apply_transform(np.linalg.inv(T_offset))
    return tmp


def get_grasp_markers(poses, scale=1.) -> List[trimesh.Trimesh]:
    markers = []
    for posemat in poses:
        marker = create_gripper_marker().apply_scale(scale)
        marker.apply_transform(posemat)
        markers.append(marker)
    return markers
