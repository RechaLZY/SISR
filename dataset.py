import glob
import h5py
import cv2
import numpy as np


# resize成可被scale整除
def Seg(image, scale=12):
    h, w = image.shape[0], image.shape[1]
    h = h // scale * scale
    w = w // scale * scale
    image = cv2.resize(image, (w, h))
    return image


# 读取图片
def Get_image(path_list, scale=12):
    image_list = []
    for i in range(len(path_list)):
        image = cv2.imread(path_list[i])
        image = Seg(image, scale)
        image_list.append(image)
    return image_list


# 训练数据切分
def Divide(image_list, scale, patch_size=41, stride=31):
    HR = []
    LR = []
    for i in range(len(image_list)):
        # 读取图片，获取Lr，Hr的Y通道图片
        image = image_list[i]
        h, w = image.shape[0], image.shape[1]
        hr = image
        lr = cv2.resize(hr, (w // scale, h // scale))
        lr = cv2.resize(lr, (w, h))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[..., 0]
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)[..., 0]

        for idx in range(0, h - patch_size, stride):
            for idy in range(0, w - patch_size, stride):
                HR.append(hr[idx:idx + patch_size, idy:idy + patch_size])
                LR.append(lr[idx:idx + patch_size, idy:idy + patch_size])

    HR = np.array(HR)
    LR = np.array(LR)
    HR = np.expand_dims(HR, axis=-1)
    LR = np.expand_dims(LR, axis=-1)

    return HR, LR


# 测试数据
def eval(image_list, scale):
    HR = []
    LR = []
    for i in range(len(image_list)):
        # 读取图片，获取Lr，Hr的Y通道图片
        image = image_list[i]
        h, w = image.shape[0], image.shape[1]
        hr = image
        lr = cv2.resize(hr, (w // scale, h // scale))
        lr = cv2.resize(lr, (w, h))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[..., 0]
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)[..., 0]

        HR.append(hr)
        LR.append(lr)

    return HR, LR


if __name__ == '__main__':

    train_path = 'D:/xm_py/x2/291/*.*'
    test_path = 'D:/xm_py/x2/Test/Set5/*.*'

    a = np.zeros((64, 32, 3))
    b = Seg(a)
    print(b.shape)

    train_image_list = Get_image(glob.glob(train_path))
    print(len(train_image_list))

    HR, LR = Divide(train_image_list, 3)
    print(HR.shape, LR.shape)

    # cv2.namedWindow('0', cv2.WINDOW_NORMAL)
    # cv2.imshow('0', HR[100])
    # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # cv2.imshow('1', LR[100])
    # cv2.waitKey(0)

    HR_2, LR_2 = Divide(train_image_list, 2)
    HR_3, LR_3 = Divide(train_image_list, 3)
    HR_4, LR_4 = Divide(train_image_list, 4)

    HR = [HR_2, HR_3, HR_4]
    LR = [LR_2, LR_3, LR_4]

    # h5_file = h5py.File('./data/train_234.h5', 'w')
    # for i in range(3):
    #     h5_file.create_dataset('hr_' + str(i + 2), data=HR[i])
    #     h5_file.create_dataset('lr_' + str(i + 2), data=LR[i])
    # h5_file.close()

    # HR = np.concatenate((HR_2, HR_3), axis=0)
    # HR = np.concatenate((HR, HR_4), axis=0)
    # LR = np.concatenate((LR_2, LR_3), axis=0)
    # LR = np.concatenate((LR, LR_4), axis=0)
    #
    # print(HR.shape, LR.shape)
    #
    # test_image_list = Get_image(glob.glob(test_path), 3)
    # HR, LR = eval(test_image_list, 3)
    # print(len(HR), len(LR))

    # h5_file = h5py.File('./data/test.h5', 'w')
    # for i in range(len(HR)):
    #     hr = HR[i]
    #     lr = LR[i]
    #     hr = np.array(hr)
    #     lr = np.array(lr)
    #     hr = np.expand_dims(hr, axis=-1)
    #     lr = np.expand_dims(lr, axis=-1)
    #     h5_file.create_dataset('hr' + str(i), data=hr)
    #     h5_file.create_dataset('lr' + str(i), data=lr)
    # h5_file.close()

