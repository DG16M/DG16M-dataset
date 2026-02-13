import os
import torch
import torch.nn as nn
import numpy as np

from se3dif import models

from se3dif.utils import get_pretrained_models_src, load_experiment_specifications
pretrained_models_dir = get_pretrained_models_src()

def load_model(args):
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications('./configs/',
                                                    load_yaml=args['pretrained_model'])
        args['classifier_path'] = model_args['classifier_path']      
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]
        args['use_attention'] = model_args['use_attention']
        args['inference_checkpoint'] = model_args['inference_checkpoint']


    if args['NetworkArch'] == 'DualGraspDiffusionConv':
        model = load_dual_arm_pointcloud_grasp_diffusion_occupancy_encoder(args, inference='pretrained_model' in args)
        print('Loaded DualGraspDiffusionConv')

    elif args['NetworkArch'] == 'DualArmPointcloudClassifier':
        model = load_dual_arm_pointcloud_classifier(args, inference='pretrained_model' in args)
        print('Loaded DualArmPointcloudClassifier')

        
    if 'pretrained_model' in args:
        model_path = args['inference_checkpoint']
        print('Loading Pretrained model from', model_path)

        ret = model.load_state_dict(torch.load(model_path), strict=False)
        print(ret)

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    return model

def load_dual_arm_pointcloud_classifier(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']


    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    plane_type = args['NetworkSpecs']['encoder']['plane_type']
    grid_resolution = args['NetworkSpecs']['encoder']['grid_resolution']
    plane_resolution = args['NetworkSpecs']['encoder']['plane_resolution']
    unet_depth = args['NetworkSpecs']['encoder']['unet_depth']

    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                grid_resolution=grid_resolution,
                                                                plane_type=plane_type,
                                                                unet=True,
                                                                plane_resolution=plane_resolution, 
                                                                device=device,
                                                                unet_depth=unet_depth).to(device)

    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    
    points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                    loc=np.array(points_params['loc']),
                                    scale=np.array(points_params['scale']))
    
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    hidden_dim = args['NetworkSpecs']['decoder']['hidden_dim']
    
    fc_classifier = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
        nn.Sigmoid()
    )
    
    model = models.ConvGraspClassifier(vision_encoder=vision_encoder,
                                       geometry_encoder=geometry_encoder,
                                       points=points,
                                       vision_decoder=vision_decoder,
                                       feature_encoder=feature_encoder,
                                       fc_classifier=fc_classifier).to(device)

    weights_path = args['NetworkSpecs']['pretrained_checkpoint']['path']
    
    if os.path.exists(weights_path) and not inference:
        model_weights = torch.load(weights_path)
        to_load = args['NetworkSpecs']['pretrained_checkpoint']['to_load']
        if 'all' in to_load:
            ret = model.load_state_dict(model_weights, strict=True)
            print(ret)
            print(f'Loaded Pretrained Weights from {weights_path}')
        elif 'none' in to_load:
            print('No Pretrained Weights Loaded')
        else:
            for name, param in model.named_parameters():
                if any([k in name for k in to_load]):
                    param.data = model_weights[name]          
            print(f'Loaded Pretrained Weights for {to_load} from {weights_path}')

    return model




def load_dual_arm_pointcloud_grasp_diffusion_occupancy_encoder(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']


    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    plane_type = args['NetworkSpecs']['encoder']['plane_type']
    grid_resolution = args['NetworkSpecs']['encoder']['grid_resolution']
    plane_resolution = args['NetworkSpecs']['encoder']['plane_resolution']
    unet_depth = args['NetworkSpecs']['encoder']['unet_depth']

    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                grid_resolution=grid_resolution,
                                                                plane_type=plane_type,
                                                                unet=True,
                                                                plane_resolution=plane_resolution, 
                                                                device=device,
                                                                unet_depth=unet_depth).to(device)

    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    
    points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                    loc=np.array(points_params['loc']),
                                    scale=np.array(points_params['scale']))
    
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    hidden_dim = args['NetworkSpecs']['decoder']['hidden_dim']

    dual_energy_net = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
    )
    
    fc_classifier = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
        nn.Sigmoid()
    )
    
    collision_predictor = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        nn.LayerNorm(hidden_dim // 2),
        nn.ELU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
        nn.LayerNorm(hidden_dim // 4),
        nn.ELU(),
        nn.Linear(hidden_dim // 4, hidden_dim//8, bias=False),
        nn.LayerNorm(hidden_dim//8),
        nn.ELU(),
        nn.Linear(hidden_dim//8, 1),
        nn.Sigmoid()
    )

    model = models.ConvGraspDiffusionFields(vision_encoder=vision_encoder, 
                                            vision_decoder=vision_decoder, 
                                            feature_encoder=feature_encoder, 
                                            geometry_encoder=geometry_encoder,
                                            decoder=dual_energy_net, points=points, 
                                            use_attention=args['use_attention'],
                                            classifier=fc_classifier,
                                            collision_predictor=collision_predictor).to(device)
    

    weights_path = args['NetworkSpecs']['pretrained_checkpoint']['path']
    
    if os.path.exists(weights_path) and not inference:
        model_weights = torch.load(weights_path)
        to_load = args['NetworkSpecs']['pretrained_checkpoint']['to_load']
        if 'all' in to_load:
            ret = model.load_state_dict(model_weights, strict=True)
            print(ret)
            print(f'Loaded Pretrained Weights from {weights_path}')
        elif 'none' in to_load:
            print('No Pretrained Weights Loaded')
        else:
            for name, param in model.named_parameters():
                if any([k in name for k in to_load]):
                    param.data = model_weights[name]          
            print(f'Loaded Pretrained Weights for {to_load} from {weights_path}')

    return model


def load_dual_arm_pointcloud_grasp_vae(args, inference=False):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    plane_type   = ['xz', 'xy', 'yz']
    grid_resolution = 32 
    plane_resolution = 32
    # k = 20 * (args['num_input_points']//1000) # scale the knn (k) by the number of input points
    vision_encoder = models.vision_encoder.VNNLocalPoolPointnet(c_dim = int(v_enc_params['latent_size'] / 3),
                                                                    grid_resolution=grid_resolution,
                                                                    plane_type=plane_type,
                                                                    unet=args['unet'],
                                                                    plane_resolution=plane_resolution, 
                                                                    device=device,
                                                                    unet_depth=5).to(device)
    vision_decoder = models.vision_encoder.LocalDecoder(c_dim = v_enc_params['latent_size'])
    # Geometry encoders
    geometry_encoder = models.geometry_encoder.map_projected_points
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    
    gaussian_mlp = nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    
    feature_decoder = nn.Sequential(
        nn.Linear(256, hidden_dim, bias=False),
        nn.LayerNorm(hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim//2, bias=False),
        nn.LayerNorm(hidden_dim//2),
        nn.ELU(),
        nn.Linear(hidden_dim//2, hidden_dim//4, bias=False),
        nn.LayerNorm(hidden_dim//4),
        nn.ELU(),
        nn.Linear(hidden_dim//4, hidden_dim//4, bias=False),
        nn.LayerNorm(hidden_dim//4),
        nn.ELU(),
        nn.Linear(hidden_dim//4, 12),
    )
    
    model = models.ConvGraspVAE(
        vision_encoder=vision_encoder, 
        geometry_encoder=geometry_encoder,
        points=points,
        vision_decoder=vision_decoder,
        feature_encoder=feature_encoder,
        gaussian_mlp=gaussian_mlp,
        feature_decoder=feature_decoder
    ).to(device=device)
    
    
    weights_path = './experiments_jul/dual_grasp_vae/checkpoints/model_current.pth'
    
    model_weights = torch.load(weights_path)
    if not inference:
        if 'model_pretrained' in weights_path:
            for name, param in model.named_parameters():
                if 'vision_encoder' in name or 'feature_encoder' in name:
                    param.data = model_weights[name]          
            print(f'Loaded Pretrained Weights for Vision Encoder and Feature Encoder from {weights_path}')
        else:
            ret = model.load_state_dict(model_weights, strict=False)
            print(ret)
            print(f'Loaded Pretrainedel_weights, Weights from {weights_path}')
        
    # print(model)
    return model