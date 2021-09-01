# coding:utf-8
import os
import numpy as np
import glob
import json


def main():
    file_dir = "/data/duser/bczhou_test/img_with_npys/"
    npy_path_list = glob.glob(os.path.join(file_dir, "*.npy"))
    for npy_path in npy_path_list:
        npy_name = os.path.basename(npy_path)
        json_name = "{}.json".format(npy_name.rsplit('.', 1)[0])
        json_path = os.path.join(file_dir, json_name)
        text_coors = np.load(npy_path).tolist()
        with open(json_path, "w") as fw:
            fw.write(json.dumps(text_coors))


if __name__ == '__main__':
    main()