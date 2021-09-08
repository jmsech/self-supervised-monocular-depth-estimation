import os

split = 'kitti'
mode = 'train'
filepath = os.path.dirname(os.path.realpath(__file__)) + \
           '/{}_{}_files.txt'.format(split, mode)
small_split = open("small_split.txt", "w+")
if split == 'kitti':
    with open(filepath, 'r') as f:
        data_list = f.read().split('\n')
        for data in data_list:
            if len(data) == 0:
                continue
            if '2011_09_26_drive_0048_sync' in data:
                small_split.write(data+'\n')