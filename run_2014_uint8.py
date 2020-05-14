# -*- coding:utf-8 -*-

import numpy as np
import cv2
import sys
import six
import os

from network_2014_uint8 import Network2014_UINT8
from chainer import datasets


class DSNTrainer:
    def __init__(self):
        self.net = Network2014_UINT8()
        
        # mnist dataset
        if not os.path.exists('mnist.npz'):
            train, test  = datasets.get_mnist()
            self.x_train = train._datasets[0]
            self.t_train = train._datasets[1]
            self.x_test  = test._datasets[0]
            self.t_test  = test._datasets[1]
            np.savez('mnist.npz', x_train=self.x_train, t_train=self.t_train, x_test=self.x_test, t_test=self.t_test)
        else:
            mnist = np.load('mnist.npz', allow_pickle=True)
            self.x_train = mnist['x_train']
            self.t_train = mnist['t_train']
            self.x_test  = mnist['x_test']
            self.t_test  = mnist['t_test']
    

    def superviser(self, iters=100, batch=10):
        # vote: (label, output): histgram of outputs
        vote = np.zeros((10, 100))
        temp_advs = np.zeros((10, 100, 1, 1, 28, 28))
        
        # voting
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='searching superviser:')
            
            # fetch mini-batch
            x = self.x_train[perm[batch*i:batch*(i+1)]]
            t = self.t_train[perm[batch*i:batch*(i+1)]]
            
            # convert input
            x = x.reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)

            # voting
            y = self.net.predict(x)
            y = y[:, 0] * 10 + y[:, 1]
            for b in six.moves.range(batch):
                vote[t[b]][y[b]] += 1.0

                # when temp_advs is not decided
                if (temp_advs[t[b]][y[b]] == 0).all():
                    temp_advs[t[b]][y[b]] = x[b].reshape(1, 1, 28, 28)

        # reps: position
        # advs: vector
        self.reps = []
        self.advs = np.zeros((10, 1, 1, 28, 28))

        for l in six.moves.range(10):
            rep = np.argmax(vote[l])
            vote[:, rep] = 0
            self.reps.append([rep//10, rep%10])
            self.advs[l] = temp_advs[l][rep]

        # duplicate check
        print(self.reps)
        seen_reps = []
        uniq_reps = [x for x in self.reps if x not in seen_reps and not seen_reps.append(x)]
        if len(self.reps) != len(uniq_reps):
            sys.exit('Error: self.reps has duplicates.')


    def validation(self, iters=100, batch=100):
        if self.reps == None:
            sys.exit('[@validation]: do superviser()')
            
        accuracy = np.zeros((10, 2))
        
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='validation:')
            
            # fetch mini-batch
            x = self.x_test[batch*i:batch*(i+1)]
            t = self.t_test[batch*i:batch*(i+1)]
            
            # convert input
            x = x.reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)

            # counting
            y = self.net.predict(x)
            for b in six.moves.range(batch):
                accuracy[t[b]][0] += 1.0 if (y[b][0], y[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else 0.0
                accuracy[t[b]][1] += 1.0
        
        print('accuracy:', np.sum(accuracy[:, 0]) / np.sum(accuracy[:, 1]))
            

    def pretraining(self, iters=400, batch=100, lr_ini=0, lr_fin=1):
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='pre-training:')

            # fetch mini-batch    
            x = self.x_train[perm[batch*i:batch*(i+1)]]
            
            # convert input
            x = x.reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)

            lr  = int(lr_ini + (lr_fin - lr_ini) * (i%(iters//4)) / (iters//4))
            stop = 1 + int(float(i) / (iters//4))
            self.net.predict(x, stop)
            self.net.pretrain(lr, stop)
            
            # imshow
            self.__wviz()

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        

    def finetuning(self, iters=600, batch=100, lr=1):
        if self.reps == None:
            sys.exit('[@finetuning]: do superviser()')
    
        perm = np.random.permutation(self.x_train.shape[0])
        for i in six.moves.range(iters):
            self.__gauge(i, iters, caption='fine-tuning:')
            
            # fetch mini-batch
            x = self.x_train[perm[batch*i:batch*(i+1)]]
            t = self.t_train[perm[batch*i:batch*(i+1)]]

            # convert input
            x = x.reshape(batch, 1, 28, 28)
            x = (x * 255).astype(np.uint8)

            # step1: normal lvq
            y = self.net.predict(x)
            m = self.__mask(y, t)
            self.net.finetune(m, lr)
            #self.__update_advs(x, t, y)

            # step2: advance prop of supervisor
            # x_adv = self.__advs(t)
            # y_adv = self.net.predict(x_adv)

            # # step3: lvq using the advance prop
            # y_useadv = self.net.predict_useadv(x)
            # m_useadv = self.__mask(y_useadv, t)
            # self.net.finetune(m_useadv, lr)


    def __mask(self, y, t):
        mask = np.zeros(y.shape[0])

        # (y == t) -> +1, (y != t) -> -1
        for b in six.moves.range(y.shape[0]):
            mask[b] = 1.0 if (y[b][0], y[b][1]) == (self.reps[t[b]][0], self.reps[t[b]][1]) else -1.0
        
        return mask


    def __advs(self, t):
        advs = []
        for l in t:
            advs.append(self.advs[l])

        return np.array(advs).reshape(t.shape[0], 1, 28, 28)


    def __update_advs(self, x, t, y):
        for l in six.moves.range(10):

            # if y include represent position
            if self.reps[l] in y:
                for i in np.where(y == self.reps[l])[0]:

                    # replace advance vector
                    if t[i] == l:
                        self.advs[l] = x[i]         


    def saveparams(self, path):
        self.net.save(path)
    

    def loadparams(self, path):
        self.net.load(path)


    def __wviz(self):
        w1 = self.net.l1[3][3].w
        img1 = np.zeros((6*10, 6*10), dtype=np.uint8)
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img1[y*6:(y+1)*6, x*6:(x+1)*6] = w1[y][x].reshape(6, 6)
        
        w2 = self.net.l2[2][2].w
        img2 = np.zeros((30*10, 30*10), dtype=np.uint8)
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img2[y*30:(y+1)*30, x*30:(x+1)*30] = w2[y][x].reshape(30, 30)

        w3 = self.net.l3[1][1].w
        img3 = np.zeros((30*10, 30*10), dtype=np.uint8)
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img3[y*30:(y+1)*30, x*30:(x+1)*30] = w3[y][x].reshape(30, 30)

        w4 = self.net.l4[0][0].w
        img4 = np.zeros((30*10, 30*10), dtype=np.uint8)
        for y in six.moves.range(10):
            for x in six.moves.range(10):
                img4[y*30:(y+1)*30, x*30:(x+1)*30] = w4[y][x].reshape(30, 30)

        img1 = cv2.resize(img1, (300, 300))
        cv2.imshow('img1@wviz', img1)
        cv2.imshow('img2@wviz', img2)
        cv2.imshow('img3@wviz', img3)
        cv2.imshow('img4@wviz', img4)
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

# pretraining
# t.pretraining()
# t.superviser()
# t.validation()
# t.saveparams('params_pretrain_0513_1.npz')

# finetuning
t.loadparams('params_finetune_0513_2.npz')
t.superviser()
t.finetuning()
t.validation()
t.saveparams('params_finetune_0513_3.npz')
