#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MyCOCO Dataset Classes

    $ python3 mycoco.py --class_type=class11

If the images and json files in the same directory, you must execute following command.
    $ python3 split_labelme2images_labesl.py --data_dir=./../echo_park --save_dir=./
    Then it will create train and val directory. In each directory, there are images and labels directory.

You must walk train and val directory and execute labelme2coco.py
    In train directory, copy classes.txt to current directory:
    $ python3 labelme2coco.py --root_dir=./ --labelme_dir=./ --save_path=./train.json
    In val directory, copy classes.txt to current directory:
    $ python3 labelme2coco.py --root_dir=./ --labelme_dir=./ --save_path=./val.json

Updated by: David
"""
import numpy as np
import os, cv2, random, time, datetime
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")


# s: single layer plate;    d: double layer plate;
PLATE_CLASSES = (  # always index 0
    's', 'd')
TRAFFIC11_CLASSES = (  # always index 0
    'person', 'bicycle', 'motor', 'tricycle', 
    'car', 'bus', 'truck', 'plate', 
    'R', 'G', 'Y')


class MyCOCO(Dataset):
    """
    Plate dataset class.
    """
    def __init__(self, 
                 img_size=640,
                 data_dir=None, 
                 image_set='train',
                 transform=None,
                 trans_config=None,
                 is_train=False,
                 load_cache=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.img_size = img_size
        self.image_set = image_set
        self.json_file = '{}.json'.format(image_set)
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(self.data_dir, image_set, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.is_train = is_train
        self.load_cache = load_cache
        # augmentation
        self.transform = transform
        self.mosaic_prob = 0
        self.mixup_prob = 0
        self.trans_config = trans_config
        if trans_config is not None:
            self.mosaic_prob = trans_config['mosaic_prob']
            self.mixup_prob = trans_config['mixup_prob']
        print('==============================')
        print('Image Set: {}'.format(image_set))
        print('Json file: {}'.format(self.json_file))
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('Ids size: {}'.format(len(self.ids)))
        print('==============================')
        # load cache data
        if load_cache:
            self._load_cache()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.pull_item(index)

    def _load_cache(self):
        # load image cache
        self.cached_images = []
        self.cached_targets = []
        dataset_size = len(self.ids)
        print('loading data into memory ...', flush=True)
        for i in range(dataset_size):
            if i % 5000 == 0:
                print("[{} / {}]".format(i, dataset_size), flush=True)
            # load an image
            image, image_id = self.pull_image(i)
            orig_h, orig_w, _ = image.shape
            # resize image
            r = self.img_size / max(orig_h, orig_w)
            if r != 1: 
                interp = cv2.INTER_LINEAR
                new_size = (int(orig_w * r), int(orig_h * r))
                image = cv2.resize(image, new_size, interpolation=interp)
            img_h, img_w = image.shape[:2]
            self.cached_images.append(image)
            # load target cache
            bboxes, labels = self.pull_anno(i)
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / orig_w * img_w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / orig_h * img_h
            self.cached_targets.append({"boxes": bboxes, "labels": labels})

    def load_image_target(self, index):
        if self.load_cache:
            # load data from cache
            image = self.cached_images[index]
            target = self.cached_targets[index]
            height, width, channels = image.shape
            target["orig_size"] = [height, width]
        else:
            # load an image
            image, _ = self.pull_image(index)
            height, width, channels = image.shape
            # load a target
            bboxes, labels = self.pull_anno(index)
            target = {
                "boxes": bboxes,
                "labels": labels,
                "orig_size": [height, width]
            }
        return image, target

    def load_mosaic(self, index):
        # load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]
        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)
        # Mosaic
        if self.trans_config['mosaic_type'] == 'yolov5_mosaic':
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config, self.is_train)
        return image, target

    def load_mixup(self, origin_image, origin_target):
        # YOLOv5 type Mixup
        if self.trans_config['mixup_type'] == 'yolov5_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            image, target = yolov5_mixup_augment(
                origin_image, origin_target, new_image, new_target)
        # YOLOX type Mixup
        elif self.trans_config['mixup_type'] == 'yolox_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
            image, target = yolox_mixup_augment(
                origin_image, origin_target, new_image, new_target, self.img_size, self.trans_config['mixup_scale'])
        return image, target
    
    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)
        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)
        # augment
        image, target, deltas = self.transform(image, target, mosaic)
        return image, target, deltas

    def pull_image(self, index):
        id_ = self.ids[index]
        im_ann = self.coco.loadImgs(id_)[0] 
        img_file = os.path.join(
                self.data_dir, self.image_set, 'images', im_ann["file_name"])
        image = cv2.imread(img_file)
        return image, id_

    def pull_anno(self, index):
        img_id = self.ids[index]
        im_ann = self.coco.loadImgs(img_id)[0]
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)
        # image infor
        width = im_ann['width']
        height = im_ann['height']
        #load a target
        bboxes = []
        labels = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                # bbox
                x1 = np.max((0, anno['bbox'][0]))
                y1 = np.max((0, anno['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                if x2 <= x1 or y2 <= y1:
                    continue
                # class label
                cls_id = self.class_ids.index(anno['category_id'])
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
        # guard against no boxes via resizing
        bboxes = np.array(bboxes).reshape(-1, 4)
        labels = np.array(labels).reshape(-1)
        return bboxes, labels


"""
    python3 traffic11.py --load_cache;sz image/*
"""
if __name__ == "__main__":
    """
    import argparse

    parser = argparse.ArgumentParser(description='FreeYOLOv2')
    # opt
    parser.add_argument('--root', default='/home/david/dataset/detect/yizhuang-COCO',
                        help='data root')
    parser.add_argument('--split', default='train',
                        help='data split')
    parser.add_argument('-size', '--img_size', default=640, type=int, 
                        help='input image size')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    parser.add_argument('--class_type', default="class11", type=str, 
                        help='class type, such as class11, palte')
    args = parser.parse_args()
    if args.class_type == "plate":
        class_type = PLATE_CLASSES
    else:
        class_type = TRAFFIC11_CLASSES

    dataset = MyCOCO(
        img_size=args.img_size,
        data_dir=args.root,
        image_set=args.split,
        transform=transform,
        trans_config=trans_config,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(30):
        image, target, deltas = dataset.pull_item(i)
        #print(target, deltas)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = class_type[cls_id]
            if x2 - x1 > 0. and y2 - y1 > 0.:
                # draw bbox
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        #cv2.imshow('gt', image)
        save_image = './images/' + str(i).zfill(4) + '.jpg'
        print('Save file: {}'.format(save_image))
        cv2.imwrite(save_image, image)
        #cv2.waitKey(0)
        """
    import torchvision
    from PIL import ImageDraw

    save_dir = "./result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 导入coco 2017 验证集和对应annotations
    coco_dataset = torchvision.datasets.CocoDetection(
                        root = "./../../datasets/yizhuang/data",
                        annFile = "./../../datasets/yizhuang/annotations/boxy_coco_train.json")
    # 图像和annotation分开读取
    image, info = coco_dataset[0]
    print(type(image))
    # ImageDraw 画图工具
    image_handler = ImageDraw.ImageDraw(image)
    #img = cv2.cvtColor(np.asarray(image_handler), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for annotation in info:
        # bbox为检测框的位置坐标
        x_min, y_min, width, height = annotation['bbox']
        # ((), ())分别为左上角的坐标对和右上角的坐标对，image_handler.rectangle是指在图片是绘制方框
        #image_handler.rectangle(((x_min, y_min), (x_min + width, y_min + height)), (0, 0, 255), 4)
        img = cv2.rectangle(img, (int(x_min), int(y_min), int(width), int(height)), (0, 0, 255), 4)
    #image.show()
    #image.save("./1.test.jpg")
    cv2.imwrite("./1.test.jpg", img)
