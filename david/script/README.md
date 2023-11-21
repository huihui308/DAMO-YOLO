
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

Labelme 格式的数据集转化为 COCO 格式的数据集
--root_dir 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
--random_split 有则会随机划分数据集，然后再分别保存为3个json文件。
--split_by_file 按照 ./train.txt ./val.txt ./test.txt 来对数据集进行划分。


Check coco dataset and annotate it on images.
```
$ rm -rf result;python3 check_coco.py --coco_dir=./../../datasets/yizhuang
```