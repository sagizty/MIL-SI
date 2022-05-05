"""
Tools   Script  ver： Apr 14th 18:50
"""
import os
import shutil
import torch
import numpy as np
from collections import OrderedDict


# Tools
def del_file(filepath):
    """
    clear all items within a folder
    :param filepath: folder path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def to_2tuple(input):
    if type(input) is tuple:
        if len(input) == 2:
            return input
        else:
            if len(input) > 2:
                output = (input[0], input[1])
                return output
            elif len(input) == 1:
                output = (input[0], input[0])
                return output
            else:
                print('cannot handle none tuple')
    else:
        if type(input) is list:
            if len(input) == 2:
                output = (input[0], input[1])
                return output
            else:
                if len(input) > 2:
                    output = (input[0], input[1])
                    return output
                elif len(input) == 1:
                    output = (input[0], input[0])
                    return output
                else:
                    print('cannot handle none list')
        elif type(input) is int:
            output = (input, input)
            return output
        else:
            print('cannot handle ', type(input))
            raise ('cannot handle ', type(input))


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
        return res

    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


# Transfer state_dict by removing misalignment
def FixStateDict(state_dict, remove_key_head=None):
    """
    Obtain a fixed state_dict by removing misalignment

    :param state_dict: model state_dict of OrderedDict()
    :param remove_key_head: the str or list of strings need to be remove by startswith
    """

    if remove_key_head is None:
        return state_dict

    elif type(remove_key_head) == str:
        keys = []
        for k, v in state_dict.items():
            if k.startswith(remove_key_head):  # 将‘arc’开头的key过滤掉，这里是要去除的层的key
                continue
            keys.append(k)

    elif type(remove_key_head) == list:
        keys = []
        for k, v in state_dict.items():
            jump = False
            for a_remove_key_head in remove_key_head:
                if k.startswith(a_remove_key_head):  # 将‘arc’开头的key过滤掉，这里是要去除的层的key
                    jump = True
                    break
            if jump:
                continue
            else:
                keys.append(k)
    else:
        print('erro in defining remove_key_head !')
        return -1

    new_state_dict = OrderedDict()
    for k in keys:
        new_state_dict[k] = state_dict[k]
    return new_state_dict


def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
