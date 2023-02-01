import h5py
import numpy as np
import tensorflow as tf
from model import VDSR
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k


def psnr(y_true, y_pred):
    return 10.0 * k.log(1.0 / (k.mean(k.square(y_pred - y_true)))) / k.log(10.0)


# 学习效率衰减
def cosine_decay(global_step):
    decay = 0.5 * (1 + np.cos(np.pi * global_step / epoch * each))   # 0.5 * (1 + (-1 - 1))
    decayed = (1 - 1e-3) * decay + 1e-3
    decayed_learning_rate = 1e-4 * decayed
    return decayed_learning_rate


model = VDSR()

# 读取数据集
train_data = h5py.File('./data/train_234.h5')
lr_2 = np.array(train_data['lr_2'])    # (None, 41, 41, 1)
hr_2 = np.array(train_data['hr_2'])    # (None, 41, 41, 1)
lr_3 = np.array(train_data['lr_3'])
hr_3 = np.array(train_data['hr_3'])
lr_4 = np.array(train_data['lr_4'])
hr_4 = np.array(train_data['hr_4'])

# 一些参数
batch = 32
epoch = 100
each = 100
num = len(lr_3) // batch

# 归一化
lr_2 = lr_2 / 127.5 - 1
hr_2 = hr_2 / 127.5 - 1
lr_3 = lr_3 / 127.5 - 1
hr_3 = hr_3 / 127.5 - 1
lr_4 = lr_4 / 127.5 - 1
hr_4 = hr_4 / 127.5 - 1

# 打乱排序
# np.random.seed(222)
# np.random.shuffle(lr_2)
# np.random.seed(222)
# np.random.shuffle(hr_2)
#
# np.random.seed(333)
# np.random.shuffle(lr_3)
# np.random.seed(333)
# np.random.shuffle(hr_3)
#
# np.random.seed(444)
# np.random.shuffle(lr_4)
# np.random.seed(444)
# np.random.shuffle(hr_4)

# lr = [lr_2, lr_3, lr_4]
# hr = [hr_2, hr_3, hr_4]

# 保存参数
# checkpoint_save_path = './VDSR_234_checkpoint/VDSR.ckpt'
# # model.load_weights(checkpoint_save_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_save_path,
#     save_weights_only=True,
#     save_best_only=False
# )

# model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#             loss=tf.keras.losses.MSE,
#             metrics=[psnr],
#         )

# for i in range(epoch):
#     loss2 = 0
#     psnr2 = 0
#     loss3 = 0
#     psnr3 = 0
#     loss4 = 0
#     psnr4 = 0
#     for j in range(num):
#         # 配置参数
#
#         # 训练
#         l2 = model.train_on_batch(lr_2[j * batch:(j + 1) * batch], hr_2[j * batch:(j + 1) * batch])
#         l3 = model.train_on_batch(lr_3[j * batch:(j + 1) * batch], hr_3[j * batch:(j + 1) * batch])
#         l4 = model.train_on_batch(lr_4[j * batch:(j + 1) * batch], hr_4[j * batch:(j + 1) * batch])
#
#         loss2 += l2[0]
#         psnr2 += l2[1]
#         loss3 += l3[0]
#         psnr3 += l3[1]
#         loss4 += l4[0]
#         psnr4 += l4[1]
#         # print(j)
#     print('*' * 50)
#     print('epoch:', i, 'scale:', 2, 'loss:', loss2 / num, 'psnr:', psnr2 / num)
#     print('epoch:', i, 'scale:', 3, 'loss:', loss3 / num, 'psnr:', psnr3 / num)
#     print('epoch:', i, 'scale:', 4, 'loss:', loss4 / num, 'psnr:', psnr4 / num)
#     print('*' * 50)
#     model.save_weights("./VDSR_234_ckpt/")


opt = tf.optimizers.Adam(1e-3)
LOSS = []
DB2 = []
DB3 = []
DB4 = []
for i in range(1, epoch+1):
    loss = 0
    dbx2 = 0
    dbx3 = 0
    dbx4 = 0
    for j in range(1, each + 1):
        lr = cosine_decay(i * j)
        opt.learning_rate.assign(lr)  # 设置学习效率
        idx = np.random.randint(0, len(lr_2), batch)
        with tf.GradientTape() as tape:
            output2 = model(lr_2[idx])
            output3 = model(lr_3[idx])
            output4 = model(lr_4[idx])
            # MSE
            loss2 = tf.reduce_mean(tf.abs(output2 - hr_2[idx]))
            loss3 = tf.reduce_mean(tf.abs(output3 - hr_3[idx]))
            loss4 = tf.reduce_mean(tf.abs(output4 - hr_4[idx]))
            cost = loss2 + loss3 + loss4
            loss += cost.numpy()

        # 梯度下降 求梯度，优化
        grads = tape.gradient(cost, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        y_pred2 = np.uint8((output2 + 1) * 255 * 0.5)
        y_true2 = np.uint8((hr_2[idx] + 1) * 255 * 0.5)

        y_pred3 = np.uint8((output3 + 1) * 255 * 0.5)
        y_true3 = np.uint8((hr_3[idx] + 1) * 255 * 0.5)

        y_pred4 = np.uint8((output4 + 1) * 255 * 0.5)
        y_true4 = np.uint8((hr_4[idx] + 1) * 255 * 0.5)

        dbx2 += tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred2, y_true2, 255), 100)).numpy()
        dbx3 += tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred3, y_true3, 255), 100)).numpy()
        dbx4 += tf.reduce_mean(tf.math.minimum(tf.image.psnr(y_pred4, y_true4, 255), 100)).numpy()

    print('epoch:', i, 'cost:', loss / each, 'db2:', dbx2 / each, 'db3:', dbx3 / each, 'db4:', dbx4 / each)
    LOSS.append(loss / each)
    DB2.append(dbx2 / each)
    DB3.append(dbx3 / each)
    DB4.append(dbx4 / each)
    model.save_weights("./VDSR_234_ckpt_11fire/")

data_save_path = 'sv11.h5'
with h5py.File(data_save_path, 'w') as hf:
    hf.create_dataset('LOSS', data=LOSS)
    hf.create_dataset('DB2', data=DB2)
    hf.create_dataset('DB3', data=DB3)
    hf.create_dataset('DB4', data=DB4)
print("save successfully")

plt.figure()
x = np.arange(epoch)
plt.plot(x, LOSS, label='loss')

plt.xlabel('Epoch')
plt.ylabel('LOSS')
plt.title("Each step's LOSS")


plt.figure()
x = np.arange(epoch)
plt.plot(x, DB2, label='db2')
plt.plot(x, DB3, label='db3')
plt.plot(x, DB4, label='db4')

plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title("Each step's PSNR")

plt.legend()
plt.show()
