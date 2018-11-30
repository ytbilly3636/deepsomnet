# -*- coding:utf-8 -*-

import numpy as np
import six
from som2d import SOM2D

class Network2014:
    def __init__(self):
        self.l1 = [[SOM2D(in_ch=1, in_size=6*6,   out_size=(10, 10)) for j in six.moves.range(7)] for i in six.moves.range(7)]
        self.l2 = [[SOM2D(in_ch=1, in_size=30*30, out_size=(10, 10)) for j in six.moves.range(5)] for i in six.moves.range(5)]
        self.l3 = [[SOM2D(in_ch=1, in_size=30*30, out_size=(10, 10)) for j in six.moves.range(3)] for i in six.moves.range(3)]
        self.l4 = [[SOM2D(in_ch=1, in_size=30*30, out_size=(10, 10)) for j in six.moves.range(1)] for i in six.moves.range(1)]

    def predict(self, x, layer=None, adv=None):
        if adv == None:
            i1 = self.__pproc_pixel(x, kernel=6, stride=4, padding=1)
            h1 = [[som.gaussian(som.forward(i1[i][j]), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l1)]
            if layer == 1:
                return 
            i2 = self.__pproc_map(h1, kernel=3, stride=1)
            h2 = [[som.gaussian(som.forward(i2[i][j]), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l2)]
            if layer == 2:
                return
            i3 = self.__pproc_map(h2, kernel=3, stride=1)
            h3 = [[som.gaussian(som.forward(i3[i][j]), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l3)]
            if layer == 3:
                return
            i4 = self.__pproc_map(h3, kernel=3, stride=1)
            h4 = [[som.forward(i4[i][j]) for j, som in enumerate(l)] for i, l in enumerate(self.l4)]
            return h4[0][0]
        
        i1 = self.__pproc_pixel(x, kernel=6, stride=4, padding=1)    
        h1 = [[som.gaussian(som.forward_adv(i1[i][j], beta=adv), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l1)]
        i2 = self.__pproc_map(h1, kernel=3, stride=1)
        h2 = [[som.gaussian(som.forward_adv(i2[i][j], beta=adv), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l2)]
        i3 = self.__pproc_map(h2, kernel=3, stride=1)
        h3 = [[som.gaussian(som.forward_adv(i3[i][j], beta=adv), var=0.4) for j, som in enumerate(l)] for i, l in enumerate(self.l3)]
        i4 = self.__pproc_map(h3, kernel=3, stride=1)
        h4 = [[som.forward_adv(i4[i][j], beta=adv) for j, som in enumerate(l)] for i, l in enumerate(self.l4)]
        return h4[0][0]
    
    def pretrain(self, lr, var, layer):
        if   layer == 1:
            [[som.update(lr, var) for som in l] for l in self.l1]
        elif layer == 2:
            [[som.update(lr, var) for som in l] for l in self.l2]
        elif layer == 3:
            [[som.update(lr, var) for som in l] for l in self.l3]
        elif layer == 4:
            [[som.update(lr, var) for som in l] for l in self.l4]

    def finetune(self, lr, mask, r=0.7):
        [[som.update_lvq(mask, lr*pow(r, 3)) for som in l] for l in self.l1]
        [[som.update_lvq(mask, lr*pow(r, 2)) for som in l] for l in self.l2]
        [[som.update_lvq(mask, lr*pow(r, 1)) for som in l] for l in self.l3]
        [[som.update_lvq(mask, lr*pow(r, 0)) for som in l] for l in self.l4]
    
    def save(self, path):
        np.savez(path, l1=self.l1, l2=self.l2, l3=self.l3, l4=self.l4)
        
    def load(self, path):
        data = np.load(path)
        self.l1 = data['l1']
        self.l2 = data['l2']
        self.l3 = data['l3']
        self.l4 = data['l4']
        
    def __pproc_pixel(self, x, kernel, stride, padding):
        if not len(x.shape) == 4:
            import sys; sys.exit('x should be BxCxHxW')
            
        pad_x = np.zeros((x.shape[0], x.shape[1], x.shape[2]+2*padding, x.shape[3]+2*padding), x.dtype)
        pad_x[:, :, padding:padding+x.shape[2], padding:padding+x.shape[3]] = x
        
        i_step = 1 + (pad_x.shape[2] - kernel) // stride
        j_step = 1 + (pad_x.shape[3] - kernel) // stride
        return [[pad_x[:, :, stride*i:stride*i+kernel, stride*j:stride*j+kernel] for j in six.moves.range(j_step)] for i in six.moves.range(i_step)]
        
    def __pproc_map(self, x, kernel, stride):
        x_h = len(x)
        x_w = len(x[0])
        c_b = x[0][0].shape[0]
        c_c = x[0][0].shape[1]
        c_h = x[0][0].shape[2]
        c_w = x[0][0].shape[3]
        whole = np.zeros((c_b, c_c, c_h*x_h, c_w*x_w))
        for i, l in enumerate(x):
            for j, chunk in enumerate(l):
                whole[:, :, c_h*i:c_h*(i+1), c_w*j:c_w*(j+1)] = chunk
                
        i_step = 1 + (x_h - kernel) // stride
        j_step = 1 + (x_w - kernel) // stride
        return [[whole[:, :, (c_h*stride)*i:(c_h*stride)*i+(c_h*kernel), (c_w*stride)*j:(c_w*stride)*j+(c_w*kernel)] for j in six.moves.range(j_step)] for i in six.moves.range(i_step)]
