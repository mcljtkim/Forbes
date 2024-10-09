import torch
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from pattern_utils import *


class BRS:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.loss_S = self.cos_loss
        self.loss_D = self.Charbonier_loss
        self.args = args
        self.softmax = torch.nn.Softmax(dim=0).to(device)
        self.upsample = torch.nn.Upsample(size=(112, 112), mode='nearest')
        self.energy_u_warp = torch.nn.MarginRankingLoss(margin=self.args.transform_margin['Warping']).to(self.device)
        self.energy_u_random_noise = torch.nn.MarginRankingLoss(margin=self.args.transform_margin['Speckle']).to(self.device)
        self.energy_c1 = torch.nn.MarginRankingLoss(margin=self.args.transform_margin['Scaling']).to(self.device)
        self.energy_c2 = torch.nn.MarginRankingLoss(margin=-1/self.args.transform_margin['Scaling']).to(self.device)
        self.transform_size = {}

    def cos_loss(self, input, target):
        loss = torch.sum(1 - torch.bmm(input.unsqueeze(1), target.unsqueeze(2)).squeeze(1).squeeze(1) / (torch.linalg.norm(input, dim=1) * torch.linalg.norm(target, dim=1)))
        return 20 * loss

    def Charbonier_loss(self, input, target):
        loss = torch.mean(torch.sqrt((input-target)**2 + 1e-6), dim=1)
        return loss

    def _size_init(self):
        b, c, h, w = self.in_img.size()
        _, c_temp, h_temp, w_temp = self.args.transform_size['Warping']
        self.transform_size['Warping'] = [b, c_temp, h_temp, w_temp]
        h_temp, w_temp, _, c_temp, _, _ = self.args.transform_size['Scaling']
        self.transform_size['Scaling'] = [h_temp, w_temp, b, c_temp, 1, 1]
        h_temp, w_temp, c_temp, _, _ = self.args.transform_size['Sinusoid']
        self.transform_size['Sinusoid'] = [h_temp, w_temp, b, c_temp, 1, 1]
        _, c_temp, h_temp, w_temp = self.args.transform_size['Speckle']
        self.transform_size['Speckle'] = [b, c_temp, h_temp, w_temp]
        _, _, h_temp, w_temp = self.args.transform_size['Averaging_blend']
        self.transform_size['Averaging_blend'] = [3, b, 3, h_temp, w_temp]
        self.transform_size['Noising_blend'] = [3, b, 3, h_temp, w_temp]

    def optimize(self, in_img):
        self.in_img = in_img
        self._size_init()
        b, c, h, w = self.in_img.shape

        # parameter initialization
        init_params = None
        bounds = []
        for key in self.args.transform_type:
            try:
                key_param, bound = generate_param(key, self.transform_size[key], in_img.device)
                bounds.extend(bound)
                if init_params == None:
                    init_params = key_param
                else:
                    init_params = torch.cat((init_params, key_param))
            except:
                pass

        in_params_ravel = init_params.detach().cpu().numpy().ravel()
        
        # obfuscating transformation and optimization
        result = fmin_l_bfgs_b(func=self._optimize_function, x0=in_params_ravel, m=20, factr=0, pgtol=1e-8, maxfun=20, maxiter=40, bounds=bounds)

        # obfuscating transformation with optimized parameters
        op_params = torch.from_numpy(result[0]).float().to(self.device)
        optimized_imgs = None
        optimized_img = in_img.clone()
        for key in self.args.transform_type:
            try:
                _len = np.prod(self.transform_size[key])
                op_param, op_params = op_params.split([_len, len(op_params) - _len])
                op_param = op_param.view(self.transform_size[key])
                if key in ['Averaging_blend', 'Noising_blend']:
                    if key == 'Averaging_blend':
                        optimized_img = torch.sum(optimized_imgs * self.upsample(self.softmax(op_param).view(b * 3, c, self.transform_size[key][-2], self.transform_size[key][-1])).view(3, b, c, h, w), dim=0)
                        optimized_imgs = None
                    else:
                        optimized_img = torch.sum(optimized_imgs * self.upsample(self.softmax(op_param).view(b * 3, c, self.transform_size[key][-2], self.transform_size[key][-1])).view(3, b, c, h, w), dim=0) + optimized_img
                else:
                    if key == 'Warping':
                        optimized_img = eval(key)(optimized_img, op_param)
                    elif key == 'Scaling':
                        optimized_img = eval(key)(optimized_img, op_param)
                    else:
                        if optimized_imgs == None:
                            optimized_imgs = eval(key)(optimized_img, op_param).unsqueeze(0)
                        else:
                            optimized_imgs = torch.cat((optimized_imgs, eval(key)(optimized_img, op_param).unsqueeze(0)))
            except:
                if optimized_imgs == None:
                    optimized_imgs = eval(key)(optimized_img, None).unsqueeze(0)
                else:
                    optimized_imgs = torch.cat((optimized_imgs, eval(key)(optimized_img, None).unsqueeze(0)))

        return optimized_img

    def _optimize_function(self, in_params):
        in_img = self.in_img
        b, c, h, w = in_img.shape
        
        x = torch.from_numpy(in_params).float().to(self.device)
        x.requires_grad_(True)

        with torch.enable_grad():
            ob_img = in_img.clone()
            ob_imgs = None
            refined_params = x
            loss = 0.0
            loss_HI = 0.0
            for key in self.args.transform_type:
                try:
                    _len = np.prod(self.transform_size[key])
                    refined_param, refined_params = refined_params.split([_len, len(refined_params) - _len])
                    refined_param = refined_param.view(self.transform_size[key])
                    if key in ['Averaging_blend', 'Noising_blend']:
                        if key == 'Averaging_blend':
                            ob_img = torch.sum(ob_imgs * self.upsample(self.softmax(refined_param).view(b * 3, c, self.transform_size[key][-2], self.transform_size[key][-1])).view(3, b, c, h, w), dim=0)
                            ob_imgs = None
                        else:
                            ob_img = torch.sum(ob_imgs * self.upsample(self.softmax(refined_param).view(b * 3, c, self.transform_size[key][-2], self.transform_size[key][-1])).view(3, b, c, h, w), dim=0) + ob_img
                    else:
                        if key == 'Warping':
                            ob_img = eval(key)(ob_img, refined_param)
                            loss_HI += self.energy_u_warp(refined_param, torch.zeros_like(refined_param, device=self.device), refined_param.sign())
                        elif key == 'Scaling':
                            ob_img = eval(key)(ob_img, refined_param)
                            param1 = refined_param[refined_param >= 1]
                            param2 = refined_param[refined_param < 1]
                            loss_HI += self.energy_c1(param1, torch.zeros_like(param1, device=self.device), param1.sign())
                            loss_HI += self.energy_c2(torch.zeros_like(param2, device=self.device), param2, param2.sign())
                        else:
                            if key == 'Speckle':
                                ob_imgs = torch.cat((ob_imgs, eval(key)(ob_img, refined_param).unsqueeze(0)))
                                loss_HI += self.energy_u_random_noise(refined_param, torch.zeros_like(refined_param, device=self.device), refined_param.sign())
                            else:
                                ob_imgs = eval(key)(ob_img, refined_param).unsqueeze(0)
                except:
                    if ob_imgs == None:
                        ob_imgs = eval(key)(ob_img, None).unsqueeze(0)
                    else:
                        ob_imgs = torch.cat((ob_imgs, eval(key)(ob_img, None).unsqueeze(0)))

            feat_x_0, feat_in_0 = map(lambda t: self.model(t)[0], (ob_img, in_img))

            loss += loss_HI
            loss += torch.sum(self.loss_D(feat_x_0, feat_in_0))
            loss += self.loss_S(feat_x_0, feat_in_0)

        f_val = loss.detach().cpu().numpy()

        loss.backward()
        f_grad = x.grad.cpu().numpy().ravel().astype(float) * b

        return [f_val, f_grad]