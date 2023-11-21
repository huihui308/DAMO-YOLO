#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://zhuanlan.zhihu.com/p/341801502

If the images and json files in different directory, you must execute following command.
    $ python3 soft_link_datasets.py --save_dir=./../../datasets/yizhuang --labelme_dir=/home/david/dataset/detect/echo_park
    Then it will create data and labels directory in '--save_dir'.

└── $ROOT_PATH
    ├── data
    └── labels

将yolo格式数据集修改成coco格式。$ROOT_PATH是根目录，需要按下面的形式组织数据：

└── $ROOT_PATH
    ├── classes.txt
    ├── data
    └── labels
classes.txt 是类的声明，一行一类。
data 目录包含所有图片 (目前支持png和jpg格式数据)
labels 目录包含所有标签(与图片同名的txt格式数据)

Copy classes.txt to current directory and execute labelme2coco.py
    $ python3 labelme2coco.py --random_split --root_dir=./../../datasets/yizhuang --labelme_dir=./../../datasets/yizhuang

配置好文件夹后，执行：python3 labelme2coco.py --root_dir $ROOT_PATH ，然后就能看见生成的 annotations 文件夹。
└── $ROOT_PATH
    ├── classes.txt
    ├── annotations
        ├── boxy_coco_train.json
        ├── boxy_coco_valid.json
        ├── boxy_coco_test.json
    ├── data
    └── labels

YOLO 格式的数据集转化为 COCO 格式的数据集
--root_dir 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
--random_split 有则会随机划分数据集，然后再分别保存为3个文件。
--split_by_file 按照 ./train.txt ./val.txt ./test.txt 来对数据集进行划分。
"""
import cv2
import json
from tqdm import tqdm
from typing import List, OrderedDict
import os, sys, signal, argparse, datetime
from sklearn.model_selection import train_test_split


def prRed(skk): print("\033[91m \r>> {}: {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prGreen(skk): print("\033[92m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prYellow(skk): print("\033[93m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightPurple(skk): print("\033[94m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prPurple(skk): print("\033[95m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prCyan(skk): print("\033[96m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightGray(skk): print("\033[97m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prBlack(skk): print("\033[98m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))


def term_sig_handler(signum, frame)->None:
    prRed('\n\n\n***************************************\n')
    prRed('catched singal: {}\n'.format(signum))
    prRed('\n***************************************\n')
    sys.stdout.flush()
    os._exit(0)


def parse_args(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
    parser.add_argument('--labelme_dir', default='./data',type=str, help="root path of images and labels for labelme")
    parser.add_argument('--save_path', type=str,default='./train.json', help="if not split the dataset, give a path to a json file")
    parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
    parser.add_argument('--split_by_file', action='store_true', help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")
    return parser.parse_args()


def get_file_list(input_dir:str, label_file_list:List[str])->None:
    imgs_list = []
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'jpg':
                #print( os.path.join(parent, filename.split('.')[0]) )
                imgs_list.append( os.path.join(parent, filename.split('.')[0]) )
    #print(imgs_list)
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'json':
                if os.path.join(parent, filename.split('.')[0]) in imgs_list:
                    label_file_list.append( os.path.join(parent, filename.split('.')[0]) )
    return


def train_test_val_split_random(img_paths,ratio_train=0.8,ratio_test=0.1,ratio_val=0.1):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_test+ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    ratio=ratio_val/(1-ratio_train)
    val_img, test_img  = train_test_split(middle_img,test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def train_test_val_split_by_files(img_paths, root_dir):
    # 根据文件 train.txt, val.txt, test.txt（里面写的都是对应集合的图片名字） 来定义训练集、验证集和测试集
    phases = ['train', 'val', 'test']
    img_split = []
    for p in phases:
        define_path = os.path.join(root_dir, f'{p}.txt')
        print(f'Read {p} dataset definition from {define_path}')
        assert os.path.exists(define_path)
        with open(define_path, 'r') as f:
            img_paths = f.readlines()
            # img_paths = [os.path.split(img_path.strip())[1] for img_path in img_paths]  # NOTE 取消这句备注可以读取绝对地址。
            img_split.append(img_paths)
    return img_split[0], img_split[1], img_split[2]


def get_class_id(class_str:str)->int:
    if class_str == 'person':
        return 0
    elif class_str == 'bicycle':
        return 1
    elif class_str in ('motor', 'motorbike'):
        return 2
    elif class_str == 'tricycle':
        return 3
    elif class_str == 'car':
        return 4
    elif class_str == 'bus':
        return 5
    elif class_str == 'truck':
        return 6
    elif class_str in ('plate', 'plate+'):
        return 7
    elif class_str == 'R':
        return 8
    elif class_str == 'G':
        return 9
    elif class_str == 'Y':
        return 10
    else:
        return -1


def labelme2coco(args):
    prGreen('args: {}'.format(args))
    root_path = args.root_dir
    prYellow('Loading data from: {}'.format(args.labelme_dir))
    assert os.path.exists(args.labelme_dir)
    #label_file_list = []
    #get_file_list(args.labelme_dir, label_file_list)
    #print( len(label_file_list) )
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    prYellow('The classes is {}'.format(classes))
    originLabelsDir = os.path.join(root_path, 'labels')
    originImagesDir = os.path.join(root_path, 'data')
    # images dir name
    indexes = os.listdir(originImagesDir)
    prGreen('indexes length: {}'.format(len(indexes)))
    if args.random_split or args.split_by_file:
        # 用于保存所有数据的图片信息和标注信息
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}
        # 建立类别标签和数字id的对应关系, 类别id从0开始。
        for (i, cls) in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        if args.random_split:
            prYellow("spliting mode: random split")
            train_img, val_img, test_img = train_test_val_split_random(indexes, 0.8, 0.1, 0.1)
        elif args.split_by_file:
            prYellow("spliting mode: split by files")
            train_img, val_img, test_img = train_test_val_split_by_files(indexes, root_path)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for (i, cls) in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    # 标注的id
    ann_id_cnt = 0
    for (k, index) in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('images','json').replace('.jpg','.json').replace('.png','.json')
        #print(txtFile)
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'data/') + index)
        height, width, _ = im.shape
        if args.random_split or args.split_by_file:
            # 切换dataset的引用对象，从而划分数据集
            if index in train_img:
                dataset = train_dataset
            elif index in val_img:
                dataset = val_dataset
            elif index in test_img:
                dataset = test_dataset
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            prRed(os.path.join(originLabelsDir, txtFile))
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            json_data = fr.read()
            parsed_json = json.loads(json_data)
            for one_obj in parsed_json['shapes']:
                #print(one_obj['label'], one_obj['points'], len(one_obj['points']), len(one_obj['points'][0]))
                if (len(one_obj['points']) != 2) or (len(one_obj['points'][0]) != 2):
                    prRed('point length {}:{} err, continue'.format(len(one_obj['points']), len(one_obj['points'][0])))
                    continue
                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = one_obj['points'][0][0]
                y1 = one_obj['points'][0][1]
                x2 = one_obj['points'][1][0]
                y2 = one_obj['points'][1][1]
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = get_class_id(one_obj['label'])
                if cls_id == -1:
                    if one_obj['label'] not in ('B'):
                        prRed('\'{}\' class not define, contine'.format(one_obj['label']))
                    #os._exit(0)
                    continue
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                #print(x1, y1, x2, y2, width, height)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1
    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.random_split or args.split_by_file:
        for phase in ['boxy_coco_train', 'boxy_coco_valid', 'boxy_coco_test']:
            json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'boxy_coco_train':
                    json.dump(train_dataset, f)
                elif phase == 'boxy_coco_valid':
                    json.dump(val_dataset, f)
                elif phase == 'boxy_coco_test':
                    json.dump(test_dataset, f)
            prYellow('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_path))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            prYellow('Save annotation to {}'.format(json_name))
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    labelme2coco(args)


if __name__ == "__main__":
    main_func()