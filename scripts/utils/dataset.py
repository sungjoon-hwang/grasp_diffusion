from pathlib import Path
import numpy as np
import sys; sys.path
import trimesh
import open3d as o3d
from typing import *
import h5py

class GraspDataset:
    def __init__(
            self,
            mesh_dir,
            grasp_data_dir,
    ):
        self.mesh_dir = Path(mesh_dir)
        self.grasp_data_dir = Path(grasp_data_dir)
        self.grasp_data_paths = list(grasp_data_dir.glob("*.h5"))

    def __len__(self):
        return len(self.grasp_data_paths)

    def get_data_info(self, index):
        path = self.grasp_data_paths[index]
        store = h5py.File(path, 'r')
        info = {}
        info['pose'] = store['grasp'][...][:, :16].reshape(-1, 4, 4)
        info['center'] = store.attrs['center']
        info['fname'] = store.attrs['fname']
        info['scale'] = store.attrs['scale']
        store.close()
        return info

    def get_signed_distance(self, mesh: trimesh.Trimesh, samples: np.ndarray):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        dist = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(samples.astype(np.float32))).numpy()
        return dist

    def get_mesh(self, info) -> trimesh.Trimesh:
        mesh_path = self.mesh_dir / f"{info['fname']}_visual.obj"
        mesh = trimesh.load(mesh_path)
        mesh.apply_translation(-info['center'])
        mesh.apply_scale(info['scale'])
        return mesh

    def __getitem__(self, index):
        info = self.get_data_info(index)
        mesh = self.get_mesh(info)

        data = {}
        data['surface_pcd'] = mesh.sample(1024)
        data['pose'] = info['pose']
        data['sdf_point_sample'] = np.random.uniform(-0.15, 0.15, size=(1000, 3))
        data['sdf_distance'] = self.get_signed_distance(mesh, data['sdf_point_sample'])
        return data