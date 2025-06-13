import os
import json
import glob
import random

import numpy as np


def load_dataset(data_dir, mode = 'train'):
     
    data_list = []

    for kimarite_dir in sorted(glob.glob(os.path.join(data_dir+'\\**\\'))):
        kimarite = os.path.basename(os.path.dirname(kimarite_dir))
        print(kimarite, end=', ')
        with open(os.path.join(kimarite_dir+kimarite+f'_{mode}.json')) as file:
            f = json.load(file)
        index = int(f['index'])-1
        print("index label:", index, end=', ')
        print("data shape", np.shape(f['data']))
        for data in f['data']:
            data_list.append({'data':data,'index':index})

    if mode == 'train':
        for _ in range(10): random.shuffle(data_list)

    data_X = []
    data_Y = []
    for data in data_list:
        data_X.append(data['data'])
        data_Y.append(data['index'])
    np.reshape(data_Y,(-1,1))

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)

    print("total data:", np.shape(data_X),np.shape(data_Y), "\n")

    return data_X, data_Y


if __name__ == '__main__':
    data_dir = "C:/Users/_s2111724/training/keypoints_dataset_by_moves_10frames"
    train_X, train_Y = load_dataset(data_dir, mode='train')
    test_X, test_Y = load_dataset(data_dir, mode='test')