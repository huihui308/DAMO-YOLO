#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Check coco dataset and annotate it on images.

    $ rm -rf result;python3 check_coco.py --coco_dir=./../../datasets/yizhuang

Updated by: David
"""
import torchvision
import numpy as np
from tqdm import tqdm
import os, cv2, random, time, signal, datetime, argparse

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
    parser.add_argument('--coco_dir', default='./data',type=str, help="Coco files directory, include ./data and ./annotations")
    parser.add_argument('--save_dir', default='./result',type=str, help="Result save directory")
    parser.add_argument('--parse_cnt', default=30,type=int, help="Parse images count")
    parser.add_argument('--class_type', default='traffic11',type=str, help="Class type, such as plate, traffic11")
    return parser.parse_args()


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)

    if args.class_type == "plate":
        class_type = PLATE_CLASSES
    else:
        class_type = TRAFFIC11_CLASSES
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    for phase in ['boxy_coco_train', 'boxy_coco_valid', 'boxy_coco_test']:
        json_name = os.path.join(args.coco_dir, 'annotations/{}.json'.format(phase))
        # 导入coco 2017 验证集和对应annotations
        coco_dataset = torchvision.datasets.CocoDetection(
                            root = os.path.join(args.coco_dir, 'data'),
                            annFile = json_name)
        # 图像和annotation分开读取
        prGreen('json_name: {}, coco_dataset length: {}'.format(json_name, len(coco_dataset)))
        for i in range(args.parse_cnt):
            coco_id = random.randint(0, len(coco_dataset))
            #prGreen('coco_id: {}'.format(coco_id))
            image, info = coco_dataset[coco_id]
            #print(type(image))
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for annotation in info:
                # bbox为检测框的位置坐标
                x1, y1, w, h = annotation['bbox']
                x1, y1 = int(x1), int(y1)
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                cls_id = int(annotation['category_id'])
                color = class_colors[cls_id]
                label = class_type[cls_id]
                #print(label)
                if ((x2 - x1) > 0) and ((y2 - y1) > 0):
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(img, label, (x1, y1 - 5), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
            save_image = os.path.join(args.save_dir, '{}_{}.jpg'.format(phase, str(coco_id).zfill(8)))
            prGreen('save_image:{}'.format(save_image))
            cv2.imwrite(save_image, img)
    return


if __name__ == "__main__":
    main_func()