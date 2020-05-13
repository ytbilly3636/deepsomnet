# -*- coding: utf-8 -*-

import numpy as np
import sys


class SOM2D_UINT8:
    def __init__(self, in_size, out_size):
        if not len(out_size) == 2:
            sys.exit('[@__init__]: out_size should be a 2-elements tuple or a list.')

        # self.w: (h, w, d)
        w = np.random.rand(out_size[0], out_size[1], in_size)
        self.w = (w * 255).astype(np.uint8)
        

    def forward(self, x, similarity=None):
        # x: (b, d)
        if not len(x.shape) == 2:
            sys.exit('[@__forward__]: x should be a (batch, dim) ndarray.')

        # reset x_useadv
        self.x_useadv = None

        # copy for buffer & update
        self.x = x.astype(np.uint8)

        # similarity
        # (b, h*w)
        if similarity == None:
            similarity = self._ip

        sim = similarity(self.x, self.w)
        sim = sim.reshape(sim.shape[0], -1)

        # winner selection
        # (b, )
        indices = np.argmax(sim, axis=1)
        cy = indices // self.w.shape[1]
        cx = indices %  self.w.shape[1]

        # (b, 2)
        self.c = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c
    

    def forward_useadv(self, x, similarity=None):
        # x: (b, d)
        if not len(x.shape) == 2:
            sys.exit('[@__forward_useadv__]: x should be a (batch, dim) ndarray.')

        # x_adv
        x_useadv = self.x * 0.5 + x * 0.5
        x_useadv[np.where(x_useadv < 0)] = 0
        x_useadv[np.where(x_useadv > 255)] = 255
        self.x_useadv = x_useadv.astype(np.uint8)

        # inner product
        # (b, d) x (h, w, d) -> (b, h, w) -> (b, h*w)
        if similarity == None:
            similarity = self._ip

        ip = similarity(self.x_useadv, self.w)
        ip = ip.reshape(ip.shape[0], -1)

        # winner selection
        # (b, )
        indices = np.argmax(ip, axis=1)
        cy = indices // self.w.shape[1]
        cx = indices %  self.w.shape[1]

        # (b, 2)
        self.c_useadv = np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        return self.c_useadv
    

    def _ip(self, x, w):
        return np.tensordot(x.astype(np.float32), w.astype(np.float32), axes=(1, 2))


    def _l2norm(self, x, w):
        # (b, d) -> (b, 1, 1, d)
        x_ = x.reshape(x.shape[0], 1, 1, -1)
        x_ = x_.astype(np.float32)

        # (h, w, d) -> (1, h, w, d)
        w_ = w.reshape((1, ) + w.shape)
        w_ = w_.astype(np.float32)

        norm = (x_ - w_) ** 2
        norm = np.sum(norm, axis=3)

        return norm * -1


    def _l1norm(self, x, w):
        # (b, d) -> (b, 1, 1, d)
        x_ = x.reshape(x.shape[0], 1, 1, -1)
        x_ = x_.astype(np.float32)

        # (h, w, d) -> (1, h, w, d)
        w_ = w.reshape((1, ) + w.shape)
        w_ = w_.astype(np.float32)

        norm = np.abs(x_ - w_)
        norm = np.sum(norm, axis=3)

        return norm * -1


    def update(self, lr):
        # h: (b, h, w) -> (b, h, w, 1)
        h = self.__shift(self.c)
        h = h.reshape(h.shape + (1, ))

        # x: (b, d) -> (b, 1, 1, d)
        x_ = self.x.reshape(self.x.shape[0], 1, 1, self.x.shape[1])
        x_ = x_.astype(np.float32)

        # w: (h, w, d) -> (1, h, w, d)
        w_ = self.w.reshape((1, ) + self.w.shape)
        w_ = w_.astype(np.float32)

        # dw: (b, h, w, d) -> (h, w, d)
        dw = (x_ - w_) * np.power(0.5, h) * np.power(0.5, lr)
        dw = np.mean(dw, axis=0)

        w_new = self.w.astype(np.float32) + dw
        w_new[np.where(w_new < 0)] = 0
        w_new[np.where(w_new > 255)] = 255
        self.w = w_new.astype(np.uint8)


    def update_useadv(self, lr, layer, layer_total):
        # layer: layer of this module
        # layer_total: total layer of this network 

        # path check: (b, ) -> (b, 1, 1, 1)
        p = self.__ccheck()
        p = p.reshape(p.shape + (1, 1, 1))
        p = p.astype(np.float32)

        # k: (b, h, w) -> (b, h, w, 1)
        k = self.__kdelta(self.c_useadv)
        k = k.reshape(k.shape + (1, ))
        k = k.astype(np.float32)

        # x: (b, c, d) -> (b, 1, 1, d)
        x_ = self.x_useadv.reshape(self.x_useadv.shape[0], 1, 1, self.x_useadv.shape[1])
        x_ = x_.astype(np.float32)

        # w: (h, w, d) -> (1, h, w, d)
        w_ = self.w.reshape((1, ) + self.w.shape)
        w_ = w_.astype(np.float32)

        # dw: (b, h, w, d) -> (h, w, d)
        dw = (x_ - w_) * p * k * np.power(0.5, lr) * np.power(0.5, layer_total - layer)
        dw = np.mean(dw, axis=0)

        w_new = self.w.astype(np.float32) + dw
        w_new[np.where(w_new < 0)] = 0
        w_new[np.where(w_new > 255)] = 255
        self.w = w_new.astype(np.uint8)
        

    def shift255(self, c):
        s = self.__shift(c)
        o = 255 >> s
        o.astype(np.uint8)
        return o
    

    def __shift(self, c):
        # c: (b, 2)
        if not len(c.shape) == 2:
            sys.exit('[@__shift__]: c should be a (batch, 2) ndarray.')

        # (b, ) -> (b, 1, 1)
        ch = c[:, 0].reshape(-1, 1, 1)
        cw = c[:, 1].reshape(-1, 1, 1)

        # meshgrid: (h, w)
        h_ax = np.arange(self.w.shape[0])   # self.w.shape[0] = h
        w_ax = np.arange(self.w.shape[1])   # self.w.shape[1] = w
        ws, hs = np.meshgrid(w_ax, h_ax)

        # (h, w) -> (1, h, w)
        hs = hs.reshape((1, ) + hs.shape)
        ws = ws.reshape((1, ) + ws.shape)

        # d: (b, h, w)
        dh = np.abs(hs - ch)
        dw = np.abs(ws - cw)
        d = dh + dw
        d[np.where(d > 8)] = 8
        d = d.astype(np.uint8)

        return d


    def __kdelta(self, c):
        # c: (b, 2)
        if not len(c.shape) == 2:
            sys.exit('[@__kdelta__]: c should be a (batch, 2) ndarray.')

        # (b, )
        ch = c[:, 0]
        cw = c[:, 1]

        # one-hot vector: (b, h), (b, w)
        oh = np.eye(self.w.shape[0])[ch]   # self.w.shape[0] = h
        ow = np.eye(self.w.shape[1])[cw]   # self.w.shape[1] = w

        # (b, h, w)
        o = oh.reshape(-1, self.w.shape[0], 1) * ow.reshape(-1, self.w.shape[1], 1)
        o.astype(np.uint8)
        return o


    def __ccheck(self):
        c = self.c[:, 0] * self.w.shape[1] + self.c[:, 1]
        c_useadv = self.c_useadv[:, 0] * self.w.shape[1] + self.c_useadv[:, 1]

        # same path -> +1, different path -> -1
        differ = (c == c_useadv)
        differ = differ.astype(np.float32)
        differ = differ * 2 - 1

        return differ


if __name__ == '__main__':
    import cv2

    # som
    som = SOM2D_UINT8(in_size=3, out_size=(10, 10))

    # learn
    ITER = 1000
    for i in range(ITER):
        color = np.random.rand(100, 3) * 255
        color = color.astype(np.uint8)

        som.forward(x=color, similarity=som._l1norm)
        som.update(lr=1)

        # check
        w = som.w
        w = cv2.resize(w, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('w', w)
        cv2.waitKey(1)