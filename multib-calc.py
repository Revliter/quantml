import numpy as np
import h5py, os
from sklearn.linear_model import LinearRegression as lr

file_000011 = []
file_000012 = []
file_000029 = []
file_000030 = []
file_000039 = []
MAGIC_COEF = -11.5348762

def walk(path):
    if not os.path.exists(path):
        print("path error")
        return -1
    for root, dirs, filenames in os.walk(path):
        if len(filenames) != 5:
            continue
        for filename in filenames:
            if str(filename) == '000011.SZtick.mat':
                file_000011.append(os.path.join(root,filename))
            elif str(filename) == '000012.SZtick.mat':
                file_000012.append(os.path.join(root,filename))
            elif str(filename) == '000029.SZtick.mat':
                file_000029.append(os.path.join(root,filename))
            elif str(filename) == '000030.SZtick.mat':
                file_000030.append(os.path.join(root,filename))
            elif str(filename) == '000039.SZtick.mat':
                file_000039.append(os.path.join(root,filename))
            else:
                print("unknown file name")
                return -1
    return 0

def calc_A(filenames, index):
    with h5py.File(filenames[index], 'r+') as f:
        BidVolume = np.array(f['r1']['BidVolume'])
        AskVolume = np.array(f['r1']['AskVolume'])

    TotalBidVolume = np.array(np.sum(BidVolume[:3:], axis=0),dtype=int)
    TotalAskVolume = np.array(np.sum(AskVolume[:3:], axis=0),dtype=int)
    miu = TotalBidVolume - TotalAskVolume

    dmiu = miu[1::] - miu[:-1:]
    return np.pad(dmiu, (0, len(miu) - len(dmiu)), 'mean')

def prune(dmiu, std):
    if len(dmiu) == std:
        return dmiu
    elif len(dmiu) > std:
        return dmiu[:std:]
    else:
        assert False

def parse_data(filenames):
    days = len(filenames)
    w_total = np.zeros((1, 5))
    for index in range(days):
        with h5py.File(filenames[index], 'r+') as f:
            Price = np.array(f['r1']['Price'])
        dPrice = Price[0][5::] - Price[0][:-5:] 
        while len(dPrice) < len(Price[0]): dPrice = np.append(dPrice, 0)
        for i in range(len(dPrice)):
            if dPrice[i] > 1: dPrice[i] = 0

        len_list = []
        dmiu_11 = calc_A(file_000011, index); len_list.append(len(dmiu_11))
        dmiu_12 = calc_A(file_000012, index); len_list.append(len(dmiu_12))
        dmiu_29 = calc_A(file_000029, index); len_list.append(len(dmiu_29))
        dmiu_30 = calc_A(file_000030, index); len_list.append(len(dmiu_30))
        dmiu_39 = calc_A(file_000039, index); len_list.append(len(dmiu_39))

        minlen = min(len_list)

        dmiu_11 = prune(dmiu_11, minlen).reshape(-1, 1)
        dmiu_12 = prune(dmiu_12, minlen).reshape(-1, 1)
        dmiu_29 = prune(dmiu_29, minlen).reshape(-1, 1)
        dmiu_30 = prune(dmiu_30, minlen).reshape(-1, 1)
        dmiu_39 = prune(dmiu_39, minlen).reshape(-1, 1)
        dPrice  = prune(dPrice , minlen).reshape(-1, 1)

        X_train = np.hstack((dmiu_11, dmiu_12, dmiu_29, dmiu_30, dmiu_39))
        Y_train = dPrice
        model = lr()
        model.fit(X_train, Y_train)
        w = model.coef_
        w_total += w

    return w_total / days


ret = walk(r'data')
if ret == -1: assert False

w_11 = parse_data(file_000011)
w_12 = parse_data(file_000012)
w_29 = parse_data(file_000029)
w_30 = parse_data(file_000030)
w_39 = parse_data(file_000039)

print(MAGIC_COEF * np.vstack((w_11, w_12, w_29, w_30, w_39)))

"""
[[ 3.28215684e-07  6.63839012e-09 -1.00268907e-08  4.48132892e-09
   8.61585966e-09]
 [ 5.07531629e-09  1.93484228e-07 -1.19245828e-08  7.13086914e-09
   4.95035541e-09]
 [-4.91166675e-09  3.77172646e-09  2.74169796e-07  7.14052929e-09
  -4.38602488e-09]
 [-2.92290470e-11  1.02314373e-09  7.92652124e-09  3.11588892e-07
   2.62929654e-09]
 [ 1.30935855e-08  2.80971602e-09  1.80795822e-09  5.21900473e-09
   3.62881350e-07]]
"""
