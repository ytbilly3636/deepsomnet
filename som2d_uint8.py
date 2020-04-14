# -*- coding: utf-8 -*-

import numpy as np
# import cupy as cp
import sys


class SOM2D_UINT8:
    def __init__(self, in_ch, in_size, out_size, gpu=False):
        if not len(out_size) == 2:
            import sys; sys.exit('out_size should be a 2-elements tuple or a list.')
        self.W = np.random.rand(out_size[0], out_size[1], in_ch, in_size)
        self.W = (self.W * 255).astype(np.uint8)
        
        # if gpu:
        #     self.W = cp.asarray(self.W)
        
    def forward(self, x):
        self.x = x.reshape(-1, self.W.shape[2], self.W.shape[3])
        ip = np.tensordot(self.x.astype(np.uint32), self.W.astype(np.uint32), axes=((1, 2), (2, 3)))
        ip = ip.reshape(ip.shape[0], -1)
        indices = np.argmax(ip, axis=1)
        cy = indices // self.W.shape[1]
        cx = indices % self.W.shape[1]
        self.c = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c
    
    '''
    def forward_gpu(self, x):       
    '''
    
    def forward_adv(self, x):
        x = x.reshape(-1, self.W.shape[2], self.W.shape[3])
        self.x = self.x * 0.5 + x * 0.5
        ip = np.tensordot(self.x.astype(np.uint32), self.W.astype(np.uint32), axes=((1, 2), (2, 3)))
        ip = ip.reshape(ip.shape[0], -1)
        indices = np.argmax(ip, axis=1)
        cy = indices // self.W.shape[1]
        cx = indices % self.W.shape[1]
        self.c = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c
    
    def update(self, lr):
        h = self.__shift(self.c)
        # (b, h, w) -> (b, h, w, 1, 1)
        h = h.reshape(h.shape + (1, 1))
        # (b, c, d) -> (b, 1, 1, c, d)
        x_ = self.x.astype(np.float32).reshape(self.x.shape[0], 1, 1, self.x.shape[1], self.x.shape[2])
        # (h, w, c, d) -> (1, h, w, c, d)
        w_ = self.W.astype(np.float32).reshape((1, ) + self.W.shape)
        dW = (x_ - w_) * np.power(0.5, h) * np.power(0.5, lr)
        self.W = self.W + np.sum(dW, axis=0) / self.x.shape[0]
        self.W = self.W.astype(np.uint8)
    
    '''    
    def update_gpu(self, lr=0.01, var=1.0):
    '''
        
    def update_lvq(self, mask, lr, l):
        h = self.__kdelta(self.c) * np.power(0.5, l)
        # (b, h, w) -> (b, h, w, 1, 1)
        h = h.reshape(h.shape + (1, 1))
        # (b, c, d) -> (b, 1, 1, c, d)
        x_ = self.x.astype(np.float32).reshape(self.x.shape[0], 1, 1, self.x.shape[1], self.x.shape[2])
        # (h, w, c, d) -> (1, h, w, c, d)
        w_ = self.W.astype(np.float32).reshape((1, ) + self.W.shape)
        dW = (x_ - w_) * np.power(0.5, h) * np.power(0.5, lr) * mask.reshape(-1, 1, 1, 1, 1)
        self.W = self.W + np.sum(dW, axis=0) / self.x.shape[0]
        self.W = self.W.astype(np.uint8)
        
    def shift255(self, c):
        s = self.__shift(c)
        o = 255 >> s
        return o.reshape(s.shape[0], 1, s.shape[1], s.shape[2]).astype(np.uint8)
    
    def __kdelta(self, c):
        indices = np.arange(c.shape[0]*self.W.shape[0]*self.W.shape[1]).reshape(c.shape[0], self.W.shape[0], self.W.shape[1])
        y = indices // self.W.shape[1] % self.W.shape[0]
        x = indices % self.W.shape[1]
        md = np.abs(y - c[:,0].reshape(-1, 1, 1)) + np.abs(x - c[:,1].reshape(-1, 1, 1))
        md[np.where(md != 0)] = -1
        return md + 1
    
    def __shift(self, c):
        indices = np.arange(c.shape[0]*self.W.shape[0]*self.W.shape[1]).reshape(c.shape[0], self.W.shape[0], self.W.shape[1])
        y = indices // self.W.shape[1] % self.W.shape[0]
        x = indices % self.W.shape[1]
        dy = np.abs(y - c[:,0].reshape(-1, 1, 1))
        dx = np.abs(x - c[:,1].reshape(-1, 1, 1))
        s = dy + dx
        s[np.where(s > 8)] = 8
        return s
