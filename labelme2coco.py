# ref https://github.com/spytensor/prepare_detection_dataset/blob/master/labelme2coco.py

import os
import json
import threading
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

np.random.seed(11631026)
random.seed(11631026)

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
        points_array = np.array(points)
        min_x = np.min(points_array[:, 0])
        min_y = np.min(points_array[:, 1])
        max_x = np.max(points_array[:, 0])
        max_y = np.max(points_array[:, 1])
        return [min_x, min_y, max_x - min_x, max_y - min_y]


def main(input_path, output_path=datetime.now().strftime("%Y%m%d_%H%M%S")):
    output_path = os.path.join("COCO_Format", output_path)
    os.makedirs(output_path, exist_ok=True)

    train_path_list = []
    val_path_list = []
    validation_ratio = 0.1

    for path in input_path:
        path = Path(path)
        json_files = list(path.glob('*.json'))
        num_files = len(json_files)

        # 計算要分配到驗證集的檔案數量
        num_validation_files = int(num_files * validation_ratio)

        # 隨機選擇檔案加入驗證集
        validation_files = random.sample(json_files, num_validation_files)

        # 將其餘的檔案加入訓練集
        train_files = [file_path for file_path in json_files if file_path not in validation_files]

        # 將路徑轉換為字串
        train_paths = [file_path.as_posix() for file_path in train_files]
        validation_paths = [file_path.as_posix() for file_path in validation_files]

        # 將檔案路徑加入對應的集合
        train_path_list.extend(train_paths)
        val_path_list.extend(validation_paths)

    print("train_number:", len(train_path_list), 'val_number:', len(val_path_list))

    def convert_train_set():
        l2c = Lableme2CoCo(classes=args.classes)
        print("Start to convert training set.")
        with open(f"{output_path}/train_path_list.txt", "w") as file:
            for path in train_path_list:
                file.write(f"{path}\n")
        train_instance = l2c.to_coco(train_path_list)
        l2c.save_coco_json(train_instance, f"{output_path}/instances_train2017.json")

    def convert_val_set():
        l2c = Lableme2CoCo(classes=args.classes)
        print("Start to convert val set.")
        with open(f"{output_path}/val_path_list.txt", "w") as file:
            for path in val_path_list:
                file.write(f"{path}\n")
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
    output_folder_name = "20231213_ValidationSet_0point1"
    main(labelme_path, output_folder_name)