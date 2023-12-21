# 使用前要先cd到coco annotation資料夾

import os
import random
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('QTAgg')  # 使用agg后端


def check_coco(type="one"):
    json_file = r"C:\NTU\Asparagus_Dataset\COCO_Format\20230726_Adam_ver_class2\instances_train2017.json"
    json_file = r"C:\NTU\Asparagus_Dataset\COCO_Format\20231213_ValidationSet_0point1\instances_val2017.json"

    coco = COCO(json_file)
    catIds = coco.getCatIds(catNms=['1','2','3','4','5']) # 不同數字表示不同类型别
    imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值

    if type == "all":
        # 全部照片檢查
        for i in range(len(imgIds)):
            img = coco.loadImgs(imgIds[i])[0] # coco dateset中image id
            img_id = img['file_name']
            I = io.imread(img_id) # 實際檔名
            plt.figure(figsize=(15,10))
            plt.axis('off')
            plt.title(img['file_name'])
            plt.imshow(I) #绘制图像，显示交给plt.show()处理
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            coco.showAnns(anns)
            plt.show() #显示图像

    elif type == "one":
        # 單張照片檢查
        img = coco.loadImgs(imgIds[random.randint(0,len(imgIds))])[0] # coco dateset中image id
        img_id = img['file_name']
        I = io.imread(img_id) # 實際檔名
        plt.figure(figsize=(15,10))
        plt.axis('off')
        plt.title(img['file_name'])
        plt.imshow(I) #绘制图像，显示交给plt.show()处理
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.show() #显示图像


if __name__ == '__main__':
    for i in range(20):
        check_coco("one")
    # check_coco("all")