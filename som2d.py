# -*- coding: utf-8 -*-

import numpy as np
#import cupy as cp

import sys
INTMAX = sys.maxsize
INTMAXINV = 1.0 / INTMAX

class SOM2D:
    def __init__(self, in_ch, in_size, out_size, gpu=False):
        if not len(out_size) == 2:
            import sys; sys.exit('out_size should be a 2-elements tuple or a list.')
        self.W = np.random.rand(out_size[0], out_size[1], in_ch, in_size)
        
        #if gpu:
        #    self.W = cp.asarray(self.W)
        
    def forward(self, x):
        self.x = x.reshape(-1, self.W.shape[2], self.W.shape[3])
        ip = np.tensordot(self.x, self.W, axes=((1, 2), (2, 3)))
        ip = ip.reshape(ip.shape[0], -1)
        #self.lc = np.max(ip, axis=1) / (np.sum(ip, axis=1) + INTMAXINV)
        #self.lc = self.lc.reshape(self.lc.shape[0], 1)
        indices = np.argmax(ip, axis=1)
        cy = indices / self.W.shape[1]
        cx = indices % self.W.shape[1]
        self.c = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c
    
    '''
    def forward_gpu(self, x):
        self.x = x.reshape(-1, self.W.shape[2], self.W.shape[3])
        ip = cp.tensordot(self.x, self.W, axes=((1, 2), (2, 3)))
        ip = ip.reshape(ip.shape[0], -1)
        #self.lc = np.max(ip, axis=1) / (np.sum(ip, axis=1) + INTMAXINV)
        #self.lc = self.lc.reshape(self.lc.shape[0], 1)
        indices = cp.argmax(ip, axis=1)
        cy = indices / self.W.shape[1]
        cx = indices % self.W.shape[1]
        self.c = cp.concatenate((cy.reshape(-1, 1), cx.reshape(-1, 1)), axis=1)
        return self.c
    '''
    
    def forward_adv(self, x, beta=0.6):
        x = x.reshape(-1, self.W.shape[2], self.W.shape[3])
        self.x = self.x * (1.0 - beta) + x * beta
        ip = np.tensordot(self.x, self.W, axes=((1, 2), (2, 3)))
        ip = ip.reshape(ip.shape[0], -1)
        self.lc = np.max(ip, axis=1) / (np.sum(ip, axis=1) + INTMAXINV)
        self.lc = self.lc.reshape(self.lc.shape[0], 1)
        indices = np.argmax(ip, axis=1)
        cy = indices / self.W.shape[1]
        cx = indices % self.W.shape[1]
        self.c = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c
    
    def update(self, lr=0.01, var=1.0):
        h = self.__gaussian(self.c, var)
        dW = np.tensordot(h, self.x, axes=(0, 0)) - self.W * np.sum(h, axis=0).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
        self.W = self.W + lr * dW / self.x.shape[0]
        self.W = self.W / np.linalg.norm(self.W.reshape(self.W.shape[0], self.W.shape[1], -1), axis=2).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
    
    '''    
    def update_gpu(self, lr=0.01, var=1.0):
        h = self.__gaussian_gpu(self.c, var)
        dW = cp.tensordot(h, self.x, axes=(0, 0)) - self.W * np.sum(h, axis=0).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
        self.W = self.W + lr * dW / self.x.shape[0]
        self.W = self.W / cp.linalg.norm(self.W.reshape(self.W.shape[0], self.W.shape[1], -1), axis=2).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
    '''
        
    def update_lvq(self, mask, lr=0.01):
        h = mask.reshape(-1, 1, 1) * self.__kdelta(self.c)
        h = h * self.__sigmoid(self.lc, mask).reshape(-1, 1, 1)
        dW = np.tensordot(h, self.x, axes=(0, 0)) - self.W * np.sum(h, axis=0).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
        self.W = self.W + lr * dW / self.x.shape[0]
        self.W = self.W / np.linalg.norm(self.W.reshape(self.W.shape[0], self.W.shape[1], -1), axis=2).reshape(self.W.shape[0], self.W.shape[1], 1, 1)
    
    def gaussian(self, c, var):
        g = self.__gaussian(c, var)
        return g.reshape(g.shape[0], 1, g.shape[1], g.shape[2])
    
    def __kdelta(self, c):
        indices = np.arange(c.shape[0]*self.W.shape[0]*self.W.shape[1]).reshape(c.shape[0], self.W.shape[0], self.W.shape[1])
        y = indices / self.W.shape[1] % self.W.shape[0]
        x = indices % self.W.shape[1]
        md = np.abs(y - c[:,0].reshape(-1, 1, 1)) + np.abs(x - c[:,1].reshape(-1, 1, 1))
        md[np.where(md != 0)] = -1
        return md + 1
                
    def __gaussian(self, c, var):
        indices = np.arange(c.shape[0]*self.W.shape[0]*self.W.shape[1]).reshape(c.shape[0], self.W.shape[0], self.W.shape[1])
        y = indices / self.W.shape[1] % self.W.shape[0]
        x = indices % self.W.shape[1]
        dy = np.abs(y - c[:,0].reshape(-1, 1, 1))
        dx = np.abs(x - c[:,1].reshape(-1, 1, 1))
        return np.exp(-(dy**2 + dx**2) / (var**2*2))
    
    '''    
    def __gaussian_gpu(self, c, var):
        indices = cp.arange(c.shape[0]*self.W.shape[0]*self.W.shape[1]).reshape(c.shape[0], self.W.shape[0], self.W.shape[1])
        y = indices / self.W.shape[1] % self.W.shape[0]
        x = indices % self.W.shape[1]
        dy = cp.abs(y - c[:,0].reshape(-1, 1, 1))
        dx = cp.abs(x - c[:,1].reshape(-1, 1, 1))
        return cp.exp(-(dy**2 + dx**2) / (var**2*2))    
    '''
        
    def __sigmoid(self, lc, mask, lam=2.0):
        return 1.0 / (1.0 + np.exp(lam * mask.reshape(mask.shape[0], 1) * (lc - (6.0 / lam))))
