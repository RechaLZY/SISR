import time

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Concatenate, Activation


class Fire(Model):
    def __init__(self, size=16):
        super(Fire, self).__init__()
        self.c1 = Conv2D(filters=size, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=size * 4, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')
        self.a2 = Activation('relu')

        self.c3 = Conv2D(filters=size * 4, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')
        self.a3 = Activation('relu')

        self.cc = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        squeeze = self.c1(inputs)
        squeeze = self.a1(squeeze)

        expand1 = self.c2(squeeze)
        expand1 = self.a2(expand1)

        expand2 = self.c3(squeeze)
        expand2 = self.a3(expand2)

        output = self.cc([expand1, expand2])
        return output


class VDSR(Model):
    def __init__(self):
        super(VDSR, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')
        self.a1 = Activation('relu')

        self.f1 = Fire()
        self.f2 = Fire()
        self.f3 = Fire()
        self.f4 = Fire()
        self.f5 = Fire()
        self.f6 = Fire()
        self.f7 = Fire()
        self.f8 = Fire()
        self.f9 = Fire()
        self.f10 = Fire()
        self.f11 = Fire()
        self.f12 = Fire()

        self.c2 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')
        self.a2 = Activation('relu')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.a1(x)

        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        x = self.f10(x)
        x = self.f11(x)
        # x = self.f12(x)

        x = self.c2(x)
        y = x + inputs
        return y


if __name__ == '__main__':
    model = VDSR()
    a = np.zeros((1, 41, 41, 1))
    for i in range(10):
        t1 = time.time()
        b = model(a)
        t2 = time.time()
        print("运行时间：", t2 - t1)
    model.summary()
