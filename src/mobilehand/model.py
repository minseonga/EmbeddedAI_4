###############################################################################
### Neural network to regress hand model from images
### Adapted from 
###     MandyMo:   https://github.com/MandyMo/pytorch_HMR
###     Boukhayma: https://github.com/boukhayma/3dhand
###############################################################################

import torch
import numpy as np 
import torch.nn as nn

from . import backbone
from .mano import MANO
from .regressor import LinearModel


class Regressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, num_param, num_iters, max_batch_size):
        super(Regressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

        self.num_param = num_param
        self.num_iters = num_iters
        mean = np.zeros(self.num_param, dtype=np.float32)
        mean_param = np.tile(mean, max_batch_size).reshape((max_batch_size, -1)) # [bs, num_param]
        self.register_buffer('mean_param', torch.from_numpy(mean_param).float())
    

    def forward(self, inputs):
        # param:
        #     inputs: is the output of ResNet encoder which has 2048 features
        # return:
        #     list of params with 3 [bs, num_param] arrays [[bs, num_param],[bs, num_param],[bs, num_param]]
        params = []
        bs     = inputs.shape[0] # Get the current batch size
        param  = self.mean_param[:bs, :] # [bs, num_param]

        for _ in range(self.num_iters):
            # Note: inputs [bs, 2048], param [bs, num_param]
            total = torch.cat([inputs, param], dim=1) # [bs, 2048 + num_param]
            param = param + self.fc_blocks(total) # [bs, num_param]
            params.append(param)

        return params


class HMR(nn.Module):
    def __init__(self, dataset='freihand'):
        super(HMR, self).__init__()

        # Number of parameters to be regressed
        # Scaling:1, Translation:2, Global rotation :3, Beta:10, Joint angle:23
        self.num_param = 39 # 1 + 2 + 3 + 10 + 23
        self.max_batch_size = 80
        self.stb_dataset = True if dataset=='stb' else False

        # Load encoder 
        self.encoder = backbone.mobilenetv3_small() # MobileNetV3
        num_features = 576
        
        # Load iterative regressor
        self.regressor = Regressor(
            fc_layers  =[num_features+self.num_param, 
                         int(num_features/2), 
                         int(num_features/2),
                         self.num_param],
            use_dropout=[True, True, False], 
            drop_prob  =[0.5, 0.5, 0], 
            use_ac_func=[True, True, False],
            num_param  =self.num_param,
            num_iters  =3,
            max_batch_size=self.max_batch_size)

        # Load MANO hand model layer
        # This will raise FileNotFoundError if MANO_RIGHT.pkl is missing
        self.mano = MANO()


    def compute_results(self, param):
        # From the input parameters [bs, num_param] 
        # Compute the resulting 2D marker location
        scale = param[:, 0].contiguous()    # [bs]    Scaling 
        trans = param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        rvec  = param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        beta  = param[:, 6:16].contiguous() # [bs,10] Shape parameters
        ang   = param[:, 16:].contiguous()  # [bs,23] Angle parameters

        pose = self.mano.convert_ang_to_pose(ang)
        vert, joint = self.mano(beta, pose, rvec)

        # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        # For STB dataset joint 0 is at palm center instead of wrist
        # Use half the distance between wrist and middle finger MCP as palm center (root joint)
        if self.stb_dataset:
            joint[:,0,:] = (joint[:,0,:] + joint[:,9,:])/2.0
        
        # Project 3D joints to 2D image using weak perspective projection 
        # only consider x and y axis so does not rely on camera intrinsic
        # [bs,21,2] * [bs,1,1] + [bs,1,2]
        keypt = joint[:,:,:2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)

        # joint = joint - joint[:,9,:].unsqueeze(1) # Make all joint relative to middle finger MCP

        return keypt, joint, vert, ang # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23]


    def forward(self, inputs, evaluation=True, get_feature=False):
        features = self.encoder(inputs)
        params   = self.regressor(features)

        if evaluation:
            # Only return final param
            keypt, joint, vert, ang = self.compute_results(params[-1])

            if get_feature:
                return keypt, joint, vert, ang, params[-1], features
            else:
                return keypt, joint, vert, ang, params[-1]
        else:
            # results  = []
            # for param in params:
            #     results.append(self.compute_results(param))
            keypt, joint, vert, ang = self.compute_results(params[-1])

            return keypt, joint, vert, ang, params # Return the list of params at each iteration
