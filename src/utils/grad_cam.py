#!/usr/bin/env python
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        # When using YoloV3 network
        one_hot = torch.FloatTensor(1,self.preds.size()[-2], self.preds.size()[-1]).zero_()
        
        # When using open-source network
        # one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()

        # print(self.preds.size())
        # print(one_hot.shape)

        # print(idx)
        #################
        flag_idx = idx[0]
        #################
        # print("Visualizing on "+str(flag_idx))
        

        # one_hot[0][idx] = 1.0
        one_hot[0][flag_idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        # print(self.idx.shape)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class BackPropagation(_PropagationBase):
    def generate(self):
        output = self.image.grad.detach().cpu().numpy()
        return output.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class Deconvolution(BackPropagation):
    def __init__(self, model):
        super(Deconvolution, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.f_maps = OrderedDict()
        self.gradient = OrderedDict()

        def forward_hook(module, input, output):
            self.f_maps[id(module)] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradient[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(forward_hook)
            module[1].register_backward_hook(backward_hook)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        L2_NORMALIZATION = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / L2_NORMALIZATION

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.f_maps, target_layer)
        grads = self._find(self.gradient, target_layer)
        weights = self._compute_grad_weights(grads)
        del grads

        grad_cam = (fmaps[0] * weights[0]).sum(dim=0)
        del fmaps
        del weights

        grad_cam = torch.clamp(grad_cam, min=0.)

        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()

        return grad_cam.detach().cpu().numpy()
