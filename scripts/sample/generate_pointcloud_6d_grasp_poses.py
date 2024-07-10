import copy
import os

import configargparse
import scipy.spatial.transform
import numpy as np
import torch

from se3dif.datasets import AcronymGraspsDirectory, AcronymGrasps
from se3dif.models.loader import load_model
from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch

def get_model(p, args, device='cpu'):

    model_params = args.model
    batch = 100
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    # ????
    context = to_torch(p[None, ...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model



def sample_pointcloud(h5_fpath):
    # acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    # mesh = acronym_grasps.avail_obj[obj_id].load_mesh()
    grasps = AcronymGrasps(h5_fpath)
    mesh = grasps.load_mesh()

    P = mesh.sample(1000)

    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()
    rot_quat = sampled_rot.as_quat()

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    translational_shift = copy.deepcopy(H)

    return P, mesh, translational_shift, rot_quat

class Args:
    def __init__(self, model='prototype', h5_fpath='path'):
        self._model = model
        self._h5_fpath = h5_fpath
    @property
    def model(self):
        return self._model

    @property
    def h5_fpath(self):
        return self._h5_fpath

    @property
    def device(self):
        return 'cuda:0'

    def __getitem__(self, item):
        return self.__dict__[item]


def generate_pcl_6d_grasp_poses(args):
    device = args.device
    h5_fpath = args.h5_fpath

    ## Set Model and Sample Generator ##
    P, mesh, trans, rot_quad = sample_pointcloud(h5_fpath)

    # generator, model = get_approximated_grasp_diffusion_field(P, args, device)
    generator, model = get_model(P, args, device)


    H = generator.sample()

    H_grasp = copy.deepcopy(H)
    # counteract the translational shift of the pointcloud (as the spawned model in simulation will still have it)
    H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1], device=device)).float()
    H[..., :3, -1] *=1/8.
    H_grasp[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh)

    print("DONE")

if __name__ == '__main__':
    args = Args(
        model='prototype',
        h5_fpath=f"{os.getenv('HOME')}/se3-diffusion/data/grasp_data/exp1_mbbchl/train/0aff119e6ce54c1db35b1608c6c1d567.h5",
    )
    generate_pcl_6d_grasp_poses(args)