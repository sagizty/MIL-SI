import os
import re
import csv
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm


def del_file(filepath):
    """
    Delete all files and folders in one directory
    :param filepath: file path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)
    # del_file(file_pack_path)


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
        print(files)
    return res


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    重命名并保存图片，生成重命名的表
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def pc_to_stander(root_from, root_to):
    root_target = root_to
    make_and_clear_path(root_target)

    f_dir_list = find_all_files(root=root_from, suffix=".png")
    print(f_dir_list)
    name_dict = {}

    for seq in tqdm(range(len(f_dir_list))):
        f_dir = f_dir_list[seq]
        f_img = Image.open(f_dir)
        img_name = os.path.split(f_dir)[1].split(".")[0]


        save_dir = os.path.join(root_target, img_name)

        name_dict[save_dir] = f_dir

        save_file(f_img, save_dir)


if __name__ == '__main__':
    pc_to_stander(root_from=r'C:\Users\86138\Desktop\MARS_Try\train\train_org_image', root_to=r"C:\Users\86138\Desktop\MARS_Try\train\train_org_image_jpg")













