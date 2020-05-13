# -*- coding:utf-8 -*-

import numpy as np
import sys
import six
from som2d_uint8 import SOM2D_UINT8


class Network2014_UINT8:
    def __init__(self):
        self.l1 = [[SOM2D_UINT8(in_size=6*6,   out_size=(10, 10)) for j in six.moves.range(7)] for i in six.moves.range(7)]
        self.l2 = [[SOM2D_UINT8(in_size=30*30, out_size=(10, 10)) for j in six.moves.range(5)] for i in six.moves.range(5)]
        self.l3 = [[SOM2D_UINT8(in_size=30*30, out_size=(10, 10)) for j in six.moves.range(3)] for i in six.moves.range(3)]
        self.l4 = [[SOM2D_UINT8(in_size=30*30, out_size=(10, 10)) for j in six.moves.range(1)] for i in six.moves.range(1)]

        # sim_i: for input layer (natural image)
        # sim_h: for hidden layer (sparse image)
        # "_ip" or "_l2norm" or "_l1norm"
        self.sim_i = SOM2D_UINT8(in_size=6*6, out_size=(10, 10))._l1norm
        self.sim_h = SOM2D_UINT8(in_size=30*30, out_size=(10, 10))._l1norm


    def predict(self, x, layer=None):
        # for pre-training or validation

        i1 = self.__pproc_pixel(x, kernel=6, stride=4, padding=1)
        h1 = [[som.shift255(som.forward(i1[i][j], self.sim_i)) for j, som in enumerate(l)] for i, l in enumerate(self.l1)]
        if layer == 1:
            return

        i2 = self.__pproc_map(h1, kernel=3, stride=1)
        h2 = [[som.shift255(som.forward(i2[i][j], self.sim_h)) for j, som in enumerate(l)] for i, l in enumerate(self.l2)]
        if layer == 2:
            return

        i3 = self.__pproc_map(h2, kernel=3, stride=1)
        h3 = [[som.shift255(som.forward(i3[i][j], self.sim_h)) for j, som in enumerate(l)] for i, l in enumerate(self.l3)]
        if layer == 3:
            return

        i4 = self.__pproc_map(h3, kernel=3, stride=1)
        h4 = [[som.forward(i4[i][j], self.sim_h) for j, som in enumerate(l)] for i, l in enumerate(self.l4)]
        return h4[0][0]


    def predict_useadv(self, x):
        # for fine tuning

        i1 = self.__pproc_pixel(x, kernel=6, stride=4, padding=1)    
        h1 = [[som.shift255(som.forward_useadv(i1[i][j], self.sim_i)) for j, som in enumerate(l)] for i, l in enumerate(self.l1)]
        
        i2 = self.__pproc_map(h1, kernel=3, stride=1)
        h2 = [[som.shift255(som.forward_useadv(i2[i][j], self.sim_h)) for j, som in enumerate(l)] for i, l in enumerate(self.l2)]
        
        i3 = self.__pproc_map(h2, kernel=3, stride=1)
        h3 = [[som.shift255(som.forward_useadv(i3[i][j], self.sim_h)) for j, som in enumerate(l)] for i, l in enumerate(self.l3)]
        
        i4 = self.__pproc_map(h3, kernel=3, stride=1)
        h4 = [[som.forward_useadv(i4[i][j], self.sim_h) for j, som in enumerate(l)] for i, l in enumerate(self.l4)]
        return h4[0][0]
    

    def pretrain(self, lr, layer):
        # layer-wised learning

        if   layer == 1:
            [[som.update(lr) for som in l] for l in self.l1]

        elif layer == 2:
            [[som.update(lr) for som in l] for l in self.l2]

        elif layer == 3:
            [[som.update(lr) for som in l] for l in self.l3]

        elif layer == 4:
            [[som.update(lr) for som in l] for l in self.l4]


    def finetune(self, lr):
        # mask: (b, ): if each batch is correct or not (+1 or -1)
        
        total = 4
        [[som.update_useadv(lr, layer=1, layer_total=total) for som in l] for l in self.l1]
        [[som.update_useadv(lr, layer=2, layer_total=total) for som in l] for l in self.l2]
        [[som.update_useadv(lr, layer=3, layer_total=total) for som in l] for l in self.l3]
        [[som.update_useadv(lr, layer=4, layer_total=total) for som in l] for l in self.l4]
    

    def save(self, path):
        np.savez(path, l1=self.l1, l2=self.l2, l3=self.l3, l4=self.l4, allow_pickle=True)
        

    def load(self, path):
        data = np.load(path, allow_pickle=True)

        self.l1 = data['l1']
        self.l2 = data['l2']
        self.l3 = data['l3']
        self.l4 = data['l4']
        

    def __pproc_pixel(self, x, kernel, stride, padding):
        if not len(x.shape) == 4:
            sys.exit('[@__pproc_pixel]: x should be BxCxHxW')
            
        b = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]

        # copy x on center of padded image
        pad_x = np.zeros((b, c, h + 2*padding, w + 2*padding), np.uint8)
        pad_x[:, :, padding:(padding + h), padding:(padding + w)] = x.astype(np.uint8)
        
        i_step = 1 + (pad_x.shape[2] - kernel) // stride
        j_step = 1 + (pad_x.shape[3] - kernel) // stride

        # each input: (b, c, h, w) -> (b, c*h*w)
        return [[pad_x[:, :, stride*i:stride*i+kernel, stride*j:stride*j+kernel].reshape(b, -1) for j in six.moves.range(j_step)] for i in six.moves.range(i_step)]
        
        
    def __pproc_map(self, x, kernel, stride):
        if not len(x[0][0].shape) == 3:
            sys.exit('[@__pproc_map]: x should be BxHxW')

        x_h = len(x)
        x_w = len(x[0])
        c_b = x[0][0].shape[0]
        c_h = x[0][0].shape[1]
        c_w = x[0][0].shape[2]

        # make whole output image
        whole = np.zeros((c_b, c_h*x_h, c_w*x_w))
        for i, l in enumerate(x):
            for j, chunk in enumerate(l):
                whole[:, c_h*i:c_h*(i+1), c_w*j:c_w*(j+1)] = chunk
                
        i_step = 1 + (x_h - kernel) // stride
        j_step = 1 + (x_w - kernel) // stride

        # each input: (b, h, w) -> (b, h*w)
        return [[whole[:, (c_h*stride)*i:(c_h*stride)*i+(c_h*kernel), (c_w*stride)*j:(c_w*stride)*j+(c_w*kernel)].reshape(c_b, -1) for j in six.moves.range(j_step)] for i in six.moves.range(i_step)]


if __name__ == '__main__':
    # net
    net = Network2014_UINT8()

    batch = 100

    # adv
    data_super = np.random.rand(batch, 1, 28, 28)
    y = net.predict(data_super)

    # use adv
    data_train = np.random.rand(batch, 1, 28, 28)
    y = net.predict_useadv(data_train)

    mask = np.random.randint(-1, 2, (batch, ))
    net.finetune(mask, lr=1)