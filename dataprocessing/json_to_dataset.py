import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
import yaml
from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', default=r'C:\Users\86138\Desktop\LYL_Part')  # 标注文件json所在的文件夹
    parser.add_argument('cell_type', default=0)
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    if args.cell_type == '0':
        cell_type = 'Negative'
    else:
        cell_type = 'Positive'

    list = os.listdir(json_file)  # 获取json文件列表
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])  # 获取每个json文件的绝对路径
        extension = list[i][-4:]
        if extension == 'json':
            filename = list[i][:-5]  # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
                # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data[
                    'shapes'])  # data['shapes']是json文件中记录着标注的位置及label等信息的字段

                # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                # lbl_viz = utils.draw.draw_label(lbl, img, captions)
                # out_dir = osp.basename(list[i])[:-5]+'_json'
                out_dir = osp.join(r'C:\Users\86138\Desktop\result\data', cell_type)
                # out_dir = r'C:\Users\86138\Desktop\LYL_Part_result\data'
                out_dir_2 = osp.join(r'C:\Users\86138\Desktop\result\mask', cell_type)
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)
                if not osp.exists(out_dir_2):
                    os.mkdir(out_dir_2)

                PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.jpg'.format(filename)))
                mask_dir = osp.join(out_dir_2, '{}.jpg'.format(filename))
                PIL.Image.fromarray(lbl).convert('RGB').save(mask_dir, quality=95)
                ff = PIL.Image.open(mask_dir)
                ff = np.array(ff) * 255
                processed_ff = PIL.Image.fromarray(np.uint8(ff))
                processed_ff.save(mask_dir)

                print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
    # python C:\Users\86138\anaconda3\envs\labelme\Scripts\labelme_json_to_dataset.exe C:\Users\86138\Desktop\All_the_data\Neg_0
