import glob
import os.path
from pathlib import Path
import numpy as np
import sys; sys.path
import trimesh
import open3d as o3d
from typing import *
import h5py
from typing import Union
from se3dif.utils import get_data_src
from torch.utils.data import DataLoader, Dataset
from se3dif.utils import to_numpy, to_torch, get_grasps_src


class GraspDirectory:
    def __init__(self, *, directory=get_grasps_src(), class_type='mug'):
        grasps_files = sorted(glob.glob(
            os.path.join(directory, class_type, 'train/*.h5')
        ))

        self.avail_obj = []
        for grasp_file in grasps_files:
            self.avail_obj.append(GraspData(grasp_file))


class GraspData:
    def __init__(self, path: Union[str, Path]):
        base_dir: str = Path(path).parents[3]  # data/

        with h5py.File(path, 'r') as store:
            self.pose = store['grasp'][...][:, :16].reshape(-1, 4, 4)
            self.center = store.attrs['center']

            # required for se3 diffusion paper
            self.mesh_scale = store.attrs['scale']
            self.mesh_id = store.attrs['fname']
            self.mesh_type = str(path).split("/")[-3]
            self.mesh_fname = f"objs/{self.mesh_type}/train/{self.mesh_id}_visual.obj"

            self.grasps, self.success = self.load_grasps()
            good_idxs = np.argwhere(self.success == 1)[:, 0]
            bad_idxs = np.argwhere(self.success == 0)[:, 0]
            self.good_grasps = self.grasps[good_idxs, ...]
            self.bad_grasps = self.grasps[bad_idxs, ...]

    # def get_signed_distance(self, mesh: trimesh.Trimesh, samples: np.ndarray):
    def get_signed_distance(self, samples: np.ndarray):
        mesh = self.load_mesh()

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        dist = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(samples.astype(np.float32))).numpy()
        return dist

    def load_grasps(self):
        # T, success
        # pose(grasp) is all success
        return self.pose, np.ones(len(self.pose))

    def get_mesh(self, *args, **kwargs) -> trimesh.Trimesh:
        return self.load_mesh()

    def load_mesh(self) -> trimesh.Trimesh:
        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')
        mesh.apply_translation(-self.center)
        mesh.apply_scale(self.mesh_scale)

        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())

        return mesh

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

    def get_signed_distance(self, data: GraspData, samples: np.ndarray):
        return data.get_signed_distance(samples)

    def get_mesh(self, data: GraspData) -> trimesh.Trimesh:
        return data.load_mesh()

    def __getitem__(self, index):
        grasp_data = GraspData(self.grasp_data_paths[index])
        mesh = self.get_mesh(grasp_data)

        data = {}
        data['surface_pcd'] = mesh.sample(1024)
        data['pose'] = grasp_data.pose
        data['sdf_point_sample'] = np.random.uniform(-0.15, 0.15, size=(1000, 3))
        data['sdf_distance'] = self.get_signed_distance(grasp_data, data['sdf_point_sample'])
        return data
