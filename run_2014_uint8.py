# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import six
import os

from network_2014_uint8 import Network2014_UINT8
from chainer import datasets

class DSNTrainer:
    def __init__(self):
        self.net = Network2014_UINT8()
        
        if not os.path.exists('mnist.npz'):
            train, test = datasets.get_mnist()
            self.x_train = train._datasets[0]
            self.t_train = train._datasets[1]
            self.x_test = test._datasets[0]
            self.t_test = test._datasets[1]
            np.savez('mnist.npz', x_train=self.x_train, t_train=self.t_train, x_test=self.x_test, t_test=self.t_test)
        else:
            mnist = np.load('mnist.npz', allow_pickle=True)
            self.x_train = mnist['x_train']
            self.t_train = mnist['t_train']
            self.x_test = mnist['x_test']
            self.t_test = mnist['t_test']
    
    def validation(self, iters=50, batch=200):
        if self.reps == None:
            sys.exit('do superviser()')
            
        accuracy = np.zeros((10, 2))
        
        indices = np.arange(self.x_test.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='validation:')
            
            x = self.x_test[indices[batch*i:batch*(i+1)]].reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)
            t = self.t_test[indices[batch*i:batch*(i+1)]]
            y = self.net.predict(x)
            
            for b in six.moves.range(batch):
                accuracy[t[b]][0] += 1.0 if (y[b][0], y[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else 0.0
                accuracy[t[b]][1] += 1.0
        
        print('accuracy:', np.sum(accuracy[:, 0]) / np.sum(accuracy[:, 1]))
            
    def pretraining(self, iters=400, batch=50, lr_ini=0, lr_fin=4):
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='pre-training:')
                
            x   = self.x_train[perm[batch*i:batch*(i+1)]].reshape(batch, 1, 28, 28)
            x   = (x * 255).astype(np.uint8)
            lr  = int(lr_ini + (lr_fin - lr_ini) * (i%(iters/4)) / (iters/4))
                
            stop = 1 + int(float(i) / (iters/4))
            self.net.predict(x, stop)
            self.net.pretrain(lr, stop)
            
            self.__wviz()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def finetuning(self, iters=600, batch=100, lr=2):
        if self.reps == None:
            sys.exit('do superviser()')
    
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='fine-tuning:')
            
            x = self.x_train[perm[batch*i:batch*(i+1)]].reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)
            t = self.t_train[perm[batch*i:batch*(i+1)]]
            y1 = self.net.predict(x)
            self.__update_advs(x, t, y1)
            y2 = self.net.predict(self.__advs(t), adv=True)
            self.net.finetune(lr, self.__mask_adv(y1, y2, t))
                
    def superviser(self, iters=50, batch=100):
        vote = np.zeros((10, 100))
        self.reps = []
        temp_advs = np.zeros((100, 1, 1, 28, 28))
        self.advs = np.zeros((10, 1, 1, 28, 28))
    
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='searching superviser:')
            
            x = self.x_train[perm[batch*i:batch*(i+1)]].reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)
            t = self.t_train[perm[batch*i:batch*(i+1)]]
            y = self.net.predict(x)
            y = y[:, 0] * 10 + y[:, 1]
            for b in six.moves.range(batch):
                vote[t[b]][y[b]] += 1.0
                if (temp_advs[y[b]] == 0).all():
                    temp_advs[y[b]] = x[b].reshape(1, 1, 28, 28)
            
        for l in six.moves.range(10):
            rep = np.argmax(vote[l])
            vote[:, rep] = 0
            self.reps.append([rep//10, rep%10])
            self.advs[l] = temp_advs[rep]

    def saveparams(self, path):
        self.net.save(path)
    
    def loadparams(self, path):
        self.net.load(path)
        
    def __mask(self, y, t):
        mask = np.zeros(y.shape[0])
        for b in six.moves.range(y.shape[0]):
            mask[b] = 1.0 if (y[b][0], y[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else -1.0
        return mask

    def __mask_adv(self, y1, y2, t):
        mask = np.zeros(t.shape[0])
        for b in six.moves.range(t.shape[0]):
            mask[b] = 1.0 if (y2[b][0], y2[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else -1.0
        for b in six.moves.range(t.shape[0]):
            mask[b] = 0.0 if (y1[b][0], y1[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else mask[b]
        return mask

    def __advs(self, t):
        advs = []
        for l in t:
            advs.append(self.advs[l])
        return np.array(advs).reshape(t.shape[0], 1, 28, 28)
        
    def __update_advs(self, x, t, y):
        for l in six.moves.range(10):
            if self.reps[l] in y:
                for i in np.where(y==self.reps[l])[0]:
                    if t[i] == l:
                        self.advs[l] = x[i] 

    def __wviz(self):
        w = self.net.l1[3][3].W
        img = np.zeros((60, 60), dtype=np.uint8)
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img[y*6:(y+1)*6, x*6:(x+1)*6] = w[y][x].reshape(6, 6)
        img = cv2.resize(img, (300, 300))
        '''
        w = self.net.l4[0][0].W
        img = np.zeros((10*30, 10*30))
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img[y*30:(y+1)*30, x*30:(x+1)*30] = w[y][x].reshape(30, 30)
        '''
        cv2.imshow('wviz', img)
        cv2.waitKey(1)

    def __gauge(self, pos, max_pos, size=20, caption='gauge:'):
        sys.stdout.write('\r' + caption + '[')
        for level in six.moves.range(size):
            if pos > (max_pos / size * level):
                sys.stdout.write('*')
            else:
                sys.stdout.write('-')
        sys.stdout.write(']')
        sys.stdout.flush()
        if pos == max_pos - 1:
            sys.stdout.write('\n')

t = DSNTrainer()

# t.pretraining()
# t.superviser()
# t.validation()
# t.saveparams('params_pretrain.npz')

t.loadparams('params_pretrain.npz')
t.superviser()
for i in six.moves.range(5):
    t.finetuning()
    t.validation()
t.saveparams('params_finetune_0415_1.npz')
