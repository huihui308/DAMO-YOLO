#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    When the images and labels in different directory, this python file can soft link these into data(images) dir and labels dir.

    $ python3 soft_link_datasets.py --save_dir=./../../datasets/yizhuang --labelme_dir=/home/david/dataset/detect/echo_park
"""
from tqdm import tqdm
from typing import List, OrderedDict
import os, sys, signal, argparse, datetime


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
    parser.add_argument('--labelme_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels")
    parser.add_argument('--save_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labelst")
    parser.add_argument('--train_ratio', default=0.8,type=float, help="Train ratio, val ratio is 1-train_ratio")
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


def soft_link_datasets(args)->None:
    prYellow('Loading data from: {}'.format(args.labelme_dir))
    assert os.path.exists(args.labelme_dir)
    train_labels_dir = os.path.join(args.save_dir, 'labels')
    train_images_dir = os.path.join(args.save_dir, 'data')
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    # get files
    label_file_list = []
    get_file_list(args.labelme_dir, label_file_list)
    prGreen( len(label_file_list) )
    for (i, header_file) in enumerate( tqdm(label_file_list) ):
        images_dir = train_images_dir
        labels_dir = train_labels_dir
        header_file_list = header_file.split('/')
        save_file_name = header_file_list[-2] + '_' + header_file_list[-1] + '.jpg'
        #print(header_file, save_file_name)
        src_file = os.path.abspath(header_file + '.jpg')
        dst_file = os.path.join(images_dir, save_file_name)
        #print(src_file, dst_file)
        os.symlink(src_file, dst_file)
        #----
        save_file_name = header_file_list[-2] + '_' + header_file_list[-1] + '.json'
        #print(header_file, save_file_name)
        src_file = os.path.abspath(header_file + '.json')
        dst_file = os.path.join(labels_dir, save_file_name)
        #print(src_file, dst_file)
        os.symlink(src_file, dst_file)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    soft_link_datasets(args)


if __name__ == "__main__":
    main_func()