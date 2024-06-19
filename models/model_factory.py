import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# from model import pointnet_cls, pointnet2_cls_msg, DetectAndVLAD,lightPNSA, lightPN,lightPNplus, pyramid, RandLANet

# from model import DetectAndVLAD, NetMG2, Backbone

from models import pptnet_v2

from models import PointNetVlad as PNV

from models.MinkLoc3dv2.layers.eca_block import ECABasicBlock
from models.MinkLoc3dv2.minkfpn import MinkFPN
from models.MinkLoc3dv2.layers.pooling_wrapper import PoolingWrapper
from models.MinkLoc3dv2.mink_params import create_resnet_block
from models.MinkLoc3dv2.minkloc import MinkLoc, MinkLocLAWS

def model_factory(args):
    #### Model
    print(args.backbone)
    if args.backbone == 'ppt':
        params = pptnet_v2.PPTparams()
        net = pptnet_v2.Network(param=params)
    if args.backbone == 'ppt_laws':
        params = pptnet_v2.PPTparams()
        net = pptnet_v2.NetworkLAWS(param=params)    
    
    if args.backbone == 'pnv':
        net = PNV.PointNetVlad(num_points=4096, 
                               global_feat=True, 
                               feature_transform=True, 
                               max_pool=False, 
                               output_dim=256,
                               use_rgb = args.use_rgb)
    if args.backbone == 'pnv_laws':
        net = PNV.PointNetVladLAWScopy(num_points=4096, 
                               global_feat=True, 
                               feature_transform=True, 
                               max_pool=False, 
                               output_dim=256,
                               use_rgb = args.use_rgb)
    if args.backbone == 'mink':
        in_channels = args.input_channel
        block_module = create_resnet_block(args.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=args.feature_size,
                          num_top_down=args.num_top_down, conv0_kernel_size=args.conv0_kernel_size,
                          block=block_module, layers=args.layers, planes=args.planes)
        pooling = PoolingWrapper(pool_method=args.pooling, in_dim=args.feature_size,
                                output_dim=args.output_dim)
        net = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=args.normalize_embeddings)
    if args.backbone == 'mink_laws':
        in_channels = args.input_channel
        block_module = create_resnet_block(args.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=args.feature_size,
                          num_top_down=args.num_top_down, conv0_kernel_size=args.conv0_kernel_size,
                          block=block_module, layers=args.layers, planes=args.planes)
        net = MinkLocLAWS(backbone=backbone, 
                            pool_method = args.pooling,
                            in_dim=args.feature_size,
                            output_dim=args.output_dim,
                            normalize_embeddings=args.normalize_embeddings)
    
    return net

