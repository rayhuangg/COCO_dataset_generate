# ref https://github.com/spytensor/prepare_detection_dataset/blob/master/labelme2coco.py

import os
import json
import threading
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

np.random.seed(77)

# 0為背景，不能使用
classname_to_id = {"stalk":1,
                   "spear":2,
                   "bar":3,
                   "straw":4,
                   "clump":5}

class Lableme2CoCo:

    def __init__(self, classes='two'):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.classes_num = classes

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)  # indent=2 縮排顯示

    # 由json文件構建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                if annotation: # 如果是不需要的類別就跳過，不是none才繼續
                    self.annotations.append(annotation)
                    self.ann_id += 1
            self.img_id += 1
        time = datetime.now().strftime("%Y%m%d_%H%M")
        instance = {}
        instance['info'] = f'RayHuang created at {time}'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 構建類別
    def _init_categories(self):
        if self.classes_num == 'two':
            selected_classes = ["stalk", "spear"]
            self.classname_to_id_selected = {k: v for k, v in classname_to_id.items() if k in selected_classes}
        elif self.classes_num == "all_without_clump":
            selected_classes = ["stalk", "spear", "straw", "bar"]
            self.classname_to_id_selected = {k: v for k, v in classname_to_id.items() if k in selected_classes}
        else:
            self.classname_to_id_selected = classname_to_id

        for k, v in self.classname_to_id_selected.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 構建COCO的image字段
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        # image['file_name'] = os.path.basename(path).replace(".json", ".jpg")  # 原始程式(只留下單一檔名，沒有路徑關係)，不推薦使用
        image['file_name'] = path.replace(".json", ".jpg")  # 取得檔案原始路徑(直接從train_path_list取得)
        return image

    # 構建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']

        if label not in self.classname_to_id_selected:
            return  None # 若類別不在指定類別中，則忽略該標註

        # 橫向X 直向Y
        if label == "clump":
            top_left_x = shape['points'][0][0]
            top_left_y = shape['points'][0][1]
            bottom_right_x = shape['points'][1][0]
            bottom_right_y = shape['points'][1][1]
            shape['points'].insert(1, [top_left_x, bottom_right_y]) # 維持標記順序，加入至原始資料的1、3項
            shape['points'].append([bottom_right_x, top_left_y])

        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 讀取json文件，返回一個json對象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 對應COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


def main(input_path, output_path=datetime.now().strftime("%Y%m%d_%H%M%S")):
    output_path = os.path.join("COCO_Format", output_path)
    os.makedirs(output_path, exist_ok=True)

    # 獲取images目錄下所有的json文件列表
    json_list_path = []
    for path in input_path:
        path = Path(path)
        json_files = path.glob('*.json')
        linux_paths = [file_path.as_posix() for file_path in json_files]
        json_list_path.extend(linux_paths)

    print(f"All data number: {len(json_list_path)}")

    train_path_list = []
    val_path_list = []

    """
    之前世鈺的validation set是從各個資料夾中選照片出來 txt檔案紀錄原始的val set清單 並且將4000~4029修改為原始對應資料 以後要新增自己的dataset可以想一下比較好的實行方法
    20230803更新: 發現之前的validation set分佈分常不均勻(Total 301)
    * Justin_labeled_data: 205
    * robot_regular_patrol/*: 96
    """

    # validation set 在這邊實際使用作為testing set
    with open('validation_list_Adam_final.txt', 'r') as file:
        validation_files = [line.strip()[:-3] for line in file]

    # 比對檔名有沒有在以前的val清單裡面，有的話就放在新的val_list裡面
    for file_path in json_list_path:
        file_name = file_path.split('/')[-1][:-4]
        if file_name in validation_files:
            val_path_list.append(file_path)
        else:
            train_path_list.append(file_path)

    print("train_number:", len(train_path_list), 'val_number:', len(val_path_list))


    # 數據劃分 80:10:10，目前沒用了
    # train_path_list, val_path_list = train_test_split(json_list_path, test_size=0.2)
    # train_path, val_path = train_test_split(train_path, test_size=0.1/0.9) # 維持比例，0.9中要保持原本的0.1部分

    def convert_train_set():
        l2c = Lableme2CoCo(classes=args.classes)
        print("Start to convert training set.")
        train_instance = l2c.to_coco(train_path_list)
        l2c.save_coco_json(train_instance, f"{output_path}/instances_train2017.json")

    def convert_val_set():
        l2c = Lableme2CoCo(classes=args.classes)
        print("Start to convert val set.")
        val_instance = l2c.to_coco(val_path_list)
        l2c.save_coco_json(val_instance, f"{output_path}/instances_val2017.json")


    # 創建多線程
    train_thread = threading.Thread(target=convert_train_set)
    val_thread = threading.Thread(target=convert_val_set)

    # 啟動線程
    train_thread.start()
    val_thread.start()

    # 等待線程结束
    train_thread.join()
    val_thread.join()

    print("Both conversions completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', choices=["all", "two", "all_without_clump"], default='two', help='選擇要轉換的類別項目:all (保留全部類別）或 two(保留spear和stalk兩類別)')
    args = parser.parse_args()

    #TODO: 原始資料檔案路徑，如有新增可以添加在這裡
    # labelme_path = ["Adam_pseudo_label/202111_patrol",
    #                 "Adam_pseudo_label/Justin_remain",
    #                 "Justin_labeled_data/",
    #                 "robot_regular_patrol/20210922_30",
    #                 "robot_regular_patrol/20211102",
    #                 "robot_regular_patrol/20211103",
    #                 "robot_regular_patrol/20211122",
    #                 "robot_regular_patrol/20211129",
    #                 "robot_regular_patrol/20211130"]

    # 1920*1080 version
    labelme_path = ["Adam_pseudo_label/202111_patrol_1920",
                    "Adam_pseudo_label/Justin_remain_1920",
                    "Justin_labeled_data_1920",
                    "robot_regular_patrol/20210922_30", # 本來就是1920*1080
                    "robot_regular_patrol/20211102",
                    "robot_regular_patrol/20211103",
                    "robot_regular_patrol/20211122",
                    "robot_regular_patrol/20211129",
                    "robot_regular_patrol/20211130"]

    # 小規模測試用(100多張)
    # labelme_path = ["robot_regular_patrol/20211130"]


    #TODO: 輸出資料夾位置，記得要修改新的資料集位置
    output_folder_name = "20231213_test_"
    main(labelme_path, output_folder_name)