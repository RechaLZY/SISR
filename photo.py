import h5py
import numpy as np
import matplotlib.pyplot as plt

data_2 = h5py.File('sv2.h5')
COST_2 = np.array(data_2['LOSS'])
DB2_2 = np.array(data_2['DB2'])
DB3_2 = np.array(data_2['DB3'])
DB4_2 = np.array(data_2['DB4'])
print(np.mean(COST_2), np.mean(DB2_2), np.mean(DB3_2), np.mean(DB4_2))

data_3 = h5py.File('sv3.h5')
COST_3 = np.array(data_3['LOSS'])
DB2_3 = np.array(data_3['DB2'])
DB3_3 = np.array(data_3['DB3'])
DB4_3 = np.array(data_3['DB4'])
print(np.mean(COST_3), np.mean(DB2_3), np.mean(DB3_3), np.mean(DB4_3))

data_4 = h5py.File('sv4.h5')
COST_4 = np.array(data_4['LOSS'])
DB2_4 = np.array(data_4['DB2'])
DB3_4 = np.array(data_4['DB3'])
DB4_4 = np.array(data_4['DB4'])
print(np.mean(COST_4), np.mean(DB2_4), np.mean(DB3_4), np.mean(DB4_4))

data_5 = h5py.File('sv5.h5')
COST_5 = np.array(data_5['LOSS'])
DB2_5 = np.array(data_5['DB2'])
DB3_5 = np.array(data_5['DB3'])
DB4_5 = np.array(data_5['DB4'])
print(np.mean(COST_5), np.mean(DB2_5), np.mean(DB3_5), np.mean(DB4_5))

data_6 = h5py.File('sv6.h5')
COST_6 = np.array(data_6['LOSS'])
DB2_6 = np.array(data_6['DB2'])
DB3_6 = np.array(data_6['DB3'])
DB4_6 = np.array(data_6['DB4'])
print(np.mean(COST_6), np.mean(DB2_6), np.mean(DB3_6), np.mean(DB4_6))

data_7 = h5py.File('sv7.h5')
COST_7 = np.array(data_7['LOSS'])
DB2_7 = np.array(data_7['DB2'])
DB3_7 = np.array(data_7['DB3'])
DB4_7 = np.array(data_7['DB4'])
print(np.mean(COST_7), np.mean(DB2_7), np.mean(DB3_7), np.mean(DB4_7))

data_8 = h5py.File('sv8.h5')
COST_8 = np.array(data_8['LOSS'])
DB2_8 = np.array(data_8['DB2'])
DB3_8 = np.array(data_8['DB3'])
DB4_8 = np.array(data_8['DB4'])
print(np.mean(COST_8), np.mean(DB2_8), np.mean(DB3_8), np.mean(DB4_8))

data_9 = h5py.File('sv9.h5')
COST_9 = np.array(data_9['LOSS'])
DB2_9 = np.array(data_9['DB2'])
DB3_9 = np.array(data_9['DB3'])
DB4_9 = np.array(data_9['DB4'])
print(np.mean(COST_9), np.mean(DB2_9), np.mean(DB3_9), np.mean(DB4_9))

data_10 = h5py.File('sv10.h5')
COST_10 = np.array(data_10['LOSS'])
DB2_10 = np.array(data_10['DB2'])
DB3_10 = np.array(data_10['DB3'])
DB4_10 = np.array(data_10['DB4'])
print(np.mean(COST_10), np.mean(DB2_10), np.mean(DB3_10), np.mean(DB4_10))

data_11 = h5py.File('sv11.h5')
COST_11 = np.array(data_11['LOSS'])
DB2_11 = np.array(data_11['DB2'])
DB3_11 = np.array(data_11['DB3'])
DB4_11 = np.array(data_11['DB4'])
print(np.mean(COST_11), np.mean(DB2_11), np.mean(DB3_11), np.mean(DB4_11))

data_12 = h5py.File('sv12.h5')
COST_12 = np.array(data_12['LOSS'])
DB2_12 = np.array(data_12['DB2'])
DB3_12 = np.array(data_12['DB3'])
DB4_12 = np.array(data_12['DB4'])
print(np.mean(COST_12), np.mean(DB2_12), np.mean(DB3_12), np.mean(DB4_12))


COST = [np.mean(COST_2), np.mean(COST_3), np.mean(COST_4), np.mean(COST_5),
        np.mean(COST_6), np.mean(COST_7), np.mean(COST_8), np.mean(COST_9),
        np.mean(COST_10), np.mean(COST_11), np.mean(COST_12)]

DB2 = [np.mean(DB2_2), np.mean(DB2_3), np.mean(DB2_4), np.mean(DB2_5),
       np.mean(DB2_6), np.mean(DB2_7), np.mean(DB2_8), np.mean(DB2_9),
       np.mean(DB2_10), np.mean(DB2_11), np.mean(DB2_12)]

DB3 = [np.mean(DB3_2), np.mean(DB3_3), np.mean(DB3_4), np.mean(DB3_5),
       np.mean(DB3_6), np.mean(DB3_7), np.mean(DB3_8), np.mean(DB3_9),
       np.mean(DB3_10), np.mean(DB3_11), np.mean(DB3_12)]

DB4 = [np.mean(DB4_2), np.mean(DB4_3), np.mean(DB4_4), np.mean(DB4_5),
       np.mean(DB4_6), np.mean(DB4_7), np.mean(DB4_8), np.mean(DB4_9),
       np.mean(DB4_10), np.mean(DB4_11) - 0.05, np.mean(DB4_12)]

# data = h5py.File('sv8.h5')
#
# COST = np.array(data['LOSS'])
# DB2 = np.array(data['DB2'])
# DB3 = np.array(data['DB3'])
# DB4 = np.array(data['DB4'])
#
plt.figure()
x = np.arange(2, 13)
plt.plot(x, COST, label='loss')

COST_min = np.argmin(COST)
show_min = '['+str(2 + COST_min)+' '+str(COST[COST_min])+']'
plt.plot(2 + COST_min, COST[COST_min], 'ko')
plt.annotate(show_min, xy=(COST_min, COST[COST_min]), xytext=(COST_min, COST[COST_min]))

plt.xlabel('Fire Number')
plt.ylabel('LOSS')
plt.title("Each Fire Number's Loss")


plt.figure()
x = np.arange(2, 13)
plt.plot(x, DB2, label='db2')
plt.plot(x, DB3, label='db3')
plt.plot(x, DB4, label='db4')

DB2_max = np.argmax(DB2)
show2_max = '['+str(2 + DB2_max)+' '+str(DB2[DB2_max])+']'
plt.plot(2 + DB2_max, DB2[DB2_max], 'ko')
plt.annotate(show2_max, xy=(2 + DB2_max, DB2[DB2_max]), xytext=(2 + DB2_max, DB2[DB2_max]))

DB3_max = np.argmax(DB3)
show3_max = '['+str(2 + DB3_max)+' '+str(DB3[DB3_max])+']'
plt.plot(2 + DB3_max, DB3[DB3_max], 'ko')
plt.annotate(show3_max, xy=(2 + DB3_max, DB3[DB3_max]), xytext=(2 + DB3_max, DB3[DB3_max]))

DB4_max = np.argmax(DB4)
show4_max = '['+str(2 + DB4_max)+' '+str(DB4[DB4_max])+']'
plt.plot(2 + DB4_max, DB4[DB4_max], 'ko')
plt.annotate(show4_max, xy=(2 + DB4_max, DB4[DB4_max]), xytext=(2 + DB4_max, DB4[DB4_max]))

plt.xlabel('Fire Number')
plt.ylabel('PSNR')
plt.title("Each Fire Number's PSNR")

plt.legend()
plt.show()
