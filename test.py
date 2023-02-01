import glob

import cv2
import h5py
import time
import math
import numpy as np
import tensorflow as tf
from model import VDSR
from dataset import Seg
import tensorflow.keras.backend as k


def psnr(y_true, y_pred):
    return 10.0 * k.log(1.0 / (k.mean(k.square(y_pred - y_true)))) / k.log(10.0)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


# test_data = h5py.File('./data/test.h5')
# for i in range(5):
#     pass
# lr = np.array(test_data['lr0'])
# hr = np.array(test_data['hr0'])

# image = cv2.imread('D:/xm_py/x2/Test/Set5/butterfly_GT.bmp')

ppp = 'D:/xm_py/x2/Test/Set14/*.bmp'
sss = 0
nnn = 0
for p in glob.glob(ppp):
    scale = 2
    image = cv2.imread(p)
    image = Seg(image, scale=scale)  # [33:198, 33:198, :]
    hr = image.copy()
    show_image = image.copy()
    h, w = hr.shape[0], hr.shape[1]

    lr = cv2.resize(hr, (w // scale, h // scale))
    lr = cv2.resize(lr, (w, h))

    YCrCb = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[..., 0]
    Cr = YCrCb[..., 1]
    Cb = YCrCb[..., 2]

    ipt = np.array(Y)
    ipt = ipt / 127.5 - 1

    model = VDSR()
    checkpoint_save_path = './VDSR_234_ckpt/'
    model.load_weights(checkpoint_save_path)

    ipt = np.expand_dims(ipt, axis=0)
    ipt = np.expand_dims(ipt, axis=-1)
    y_pred = model.predict(ipt)
    y_pred = np.reshape(y_pred, (h, w))
    y_pred = np.uint8((y_pred + 1) * 255 * 0.5)

    vdsr = cv2.merge([y_pred, Cr, Cb])
    vdsr = cv2.cvtColor(vdsr, cv2.COLOR_YCrCb2BGR)

    # cv2.imshow('image', show_image)
    # cv2.imshow('lr', lr)
    # cv2.imshow('lr(bicubic)', lr)
    # cv2.imshow('vdsr', vdsr)
    # cv2.waitKey(0)

    print('----psnr----')
    print('lr(bicubic):', PSNR(lr, hr))  # 32.30929590648262
    print('vdsr:', PSNR(vdsr, hr))  # 33.63005892806318
    sss += PSNR(vdsr, hr)
    nnn += 1
print('mean=', sss / nnn)


# # image = cv2.imread('E:/data_2017/20221123/data_new/003/(0_4.bmp')
# scale = 3
# image = Seg(image, scale=scale)        # [33:198, 33:198, :]
# # hr = image[33:66, 33:66, :]
# hr = image.copy()
# # show_image = cv2.rectangle(image.copy(), (33, 33), (66, 66), (0, 0, 255), 3)
# show_image = image.copy()
# h, w = hr.shape[0], hr.shape[1]
#
# lr = cv2.resize(hr, (w // scale, h // scale))
# lr = cv2.resize(lr, (w, h))
#
# # lr = cv2.resize(hr, (w * scale, h * scale))
# # h, w = lr.shape[0], lr.shape[1]
#
# YCrCb = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)
# Y = YCrCb[..., 0]
# Cr = YCrCb[..., 1]
# Cb = YCrCb[..., 2]
#
# ipt = np.array(Y)
# ipt = ipt / 127.5 - 1
#
# model = VDSR()
# checkpoint_save_path = './VDSR_234_ckpt/'
# model.load_weights(checkpoint_save_path)
# # model = tf.keras.models.load_model("./trained_vdsr_model", custom_objects={'psnr': psnr})
#
# ipt = np.expand_dims(ipt, axis=0)
# ipt = np.expand_dims(ipt, axis=-1)
# y_pred = model.predict(ipt)
# y_pred = np.reshape(y_pred, (h, w))
# y_pred = np.uint8((y_pred + 1) * 255 * 0.5)
#
# vdsr = cv2.merge([y_pred, Cr, Cb])
# vdsr = cv2.cvtColor(vdsr, cv2.COLOR_YCrCb2BGR)
#
# cv2.imshow('image', show_image)
# cv2.imshow('hr', hr)
# cv2.imshow('lr', lr)
# cv2.imshow('lr(bicubic)', lr)
# cv2.imshow('vdsr', vdsr)
# cv2.waitKey(0)
#
# print('----psnr----')
# print('lr(bicubic):', PSNR(lr, hr))   # 32.30929590648262
# print('vdsr:', PSNR(vdsr, hr))     # 33.63005892806318
#
# for i in range(10):       # time= 0.023000478744506836
#     t1 = time.time()
#     t = model.predict(ipt)
#     t2 = time.time()
#     print('time=', t2 - t1)

"""
baby: lr(bicubic): 35.71988098387658, vdsr: 36.77170576556549
bird:lr(bicubic): 35.44315193565418, vdsr: 37.10826480920078
butterfly:lr(bicubic): 32.0372908948357, vdsr: 33.60988013521303
head:lr(bicubic): 32.64728135302978, vdsr: 32.94788224300223
woman:lr(bicubic): 34.037431081066416, vdsr: 35.45372801561125
mean_x2:33.9768, 35.1782

baby: lr(bicubic): 35.384646159256576, vdsr: 35.6720050965685
bird:lr(bicubic): 34.283884827824885, vdsr: 34.76259597933123
butterfly:lr(bicubic): 32.1523020524367, vdsr: 33.119370764354684
head:lr(bicubic): 33.83583314982411, vdsr: 33.82024253260803
woman:lr(bicubic): 33.62636663277777, vdsr: 34.179594525296565
mean_x3:33.8566, 34.301

baby: lr(bicubic): 33.24179110113688, vdsr: 33.60812647976314
bird:lr(bicubic): 32.44168725652558, vdsr:32.97499842629259
butterfly:lr(bicubic): 30.47773509872712, vdsr: 31.357836423993003
head:lr(bicubic): 31.632035271462904, vdsr: 31.694739912757694
woman:lr(bicubic): 32.199426938354, vdsr: 32.82458928804184
mean_x4:31.9988, 32.4922
"""

"""
x2: 32.8
x3: 33.0
x4: 33.2
x5: 33.6
x6: 33.5
x7: 33.4
x8: 33.1
"""

