import numpy as np
import h5py, os

file_000011 = []
file_000012 = []
file_000029 = []
file_000030 = []
file_000039 = []

def walk(path):
    if not os.path.exists(path):
        print("path error")
        return -1
    for root, dirs, filenames in os.walk(path):
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

ret = walk(r'data')
if ret == -1:
    assert False
print(len(file_000011))
