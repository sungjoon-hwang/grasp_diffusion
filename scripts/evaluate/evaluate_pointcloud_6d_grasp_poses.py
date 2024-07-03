# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']
import os.path


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=12)
    p.add_argument('--n_grasps', type=str, default='1000')
    p.add_argument('--n_envs', type=str, default='20')
    p.add_argument('--obj_class', type=str, default='mug')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--eval_sim', type=bool, default=True)
    p.add_argument('--model', type=str, default='grasp_dif_multi')

    opt = p.parse_args()
    return opt


def get_model(args, device='cpu'):
    model_params = args.model
    batch = 100
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model


if __name__ == '__main__':
    import copy
    # import isaacgym
    import configargparse
    import torch
    args = parse_args()
    from se3dif.models.loader import load_model
    from se3dif.models import loader
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    import json

    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = int(args.n_envs)
    device = args.device


    base = '/home/irsl/24-se3-df/'
    param_path = os.path.join(base, 'grasp_duffusion/scripts/train/params/multiobject_p_graspdif/params.json')
    model_path = os.path.join(base, 'logs/multiobject_p_graspdif/checkpoints/model_epoch_1660_iter_044820.pth')

    model_args = json.load(open(param_path))
    model_args['saving_folder'] = os.path.join(base, 'logs/multiobject_p_graspdif')
    model_args['device'] = device

    ## Get Model and Sample Generator ##
    # generator, model = get_model(args, device)
    model = loader.load_model(model_args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator = Grasp_AnnealedLD(model, batch=100, T=70, T_fit=50, k_steps=2, device=device)

    #### Build Model Generator ####
    from isaac_evaluation.grasp_quality_evaluation.evaluate_model import EvaluatePointConditionedGeneratedGrasps
    evaluator = EvaluatePointConditionedGeneratedGrasps(generator, n_grasps=n_grasps, obj_id=obj_id, obj_class=obj_class, n_envs=n_envs,
                                                        viewer=True)

    mesh = ''

    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()
    rot_quat = sampled_rot.as_quat()

    P = mesh.sample(1000)
    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(success_eval=False, earth_moving_distance=False)

