# source https://blog.csdn.net/gaoyi135/article/details/110466932
import cv2
import os
import glob
import json
import collections
import numpy as np
from labelme import utils
from tqdm import tqdm


if __name__ == "__main__":
    # src_dir = r'C:\NTU\Asparagus_Dataset\Adam_pseudo_label\202111_patrol'
    # src_dir = r'C:\NTU\Asparagus_Dataset\Adam_pseudo_label\Justin_remain'
    src_dir = r'C:\NTU\Asparagus_Dataset\Justin_labeled_data'

    dst_dir = src_dir + "_1920"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 先收集一下文件夾中圖片的格式列表，例如 ['.jpg', '.JPG']
    exts = dict()
    filesnames = os.listdir(src_dir)
    for filename in filesnames:
        name, ext = filename.split('.')
        if ext != 'json':
            if exts.__contains__(ext):
                exts[ext] += 1
            else:
                exts[ext] = 1

    anno = collections.OrderedDict()  # 使用有序字典以確保保存和讀取的字典順序一致，避免使用 dict() 可能導致順序錯亂
    for key in exts.keys():
        for img_file in tqdm(glob.glob(os.path.join(src_dir, '*.' + key))):
            file_name = os.path.basename(img_file)
            print(f"Processing {file_name}")
            img = cv2.imread(img_file)
            (h, w, c) = img.shape
            w_new = 1920  # 指定的最長邊
            h_new = int(h / w * w_new)  # 高度等比例縮放
            ratio = w_new / w  # 標注文件中的座標乘以這個比例可得到新的座標值
            img_resize = cv2.resize(img, (w_new, h_new))  # resize 中的目標尺寸參數為 (寬度, 高度)
            cv2.imwrite(os.path.join(dst_dir, file_name), img_resize)

            # 處理標註文件 JSON 中的標註點的縮放
            json_file = os.path.join(src_dir, file_name.split('.')[0] + '.json')
            save_to = open(os.path.join(dst_dir, file_name.split('.')[0] + '.json'), 'w')
            with open(json_file, 'rb') as f:
                anno = json.load(f)
                for shape in anno["shapes"]:
                    points = shape["points"]
                    points = (np.array(points) * ratio).tolist()
                    shape["points"] = points

                # 注意下面的 img_resize 編碼加密之前要記得將通道順序由 BGR 變回 RGB
                anno['imageData'] = str(utils.img_arr_to_b64(img_resize[..., (2, 1, 0)]), encoding='utf-8')
                json.dump(anno, save_to, indent=2)
    print("Done")
