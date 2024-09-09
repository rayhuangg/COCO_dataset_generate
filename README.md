用途：轉檔Labelme標記的Instanse Segmentation格式，輸出為CoCo Dataset資料集格式，以符合Detectron2的資料註冊方法


## 轉檔機制設計核心想法

### 原本的管理資料集方式

* train/val 資料夾人工分配照片、標記檔案
* 不好擴充新增的照片
* 有新增檔案就會破壞原本的資料集，無法保留記錄
* 同樣資料照片會儲存相當多份，散佈在硬碟不同位置中(原始資料, dataset v1, dataset v2… )

### 修正方向

* **最重要假設：單位資料夾內的資料分佈是均勻相似的，才可以做隨機分配**
* 照片維持在各自的原始資料夾，可以有意義的記錄資料夾名稱，用相對路徑管理照片路徑，維持資料夾架構就可以持續擴充
* 最小單位資料夾，固定比例切分資料集。
* random seed 保證隨機分配可以被重現，新增的資料夾內容後仍維持先前已隨機分配的照片分佈
* 照片資料不需重新複製貼上，方便管理不同版本，照片只會存在一次
* 原先部分照片有標記水管、吸管、叢集，在轉檔時先將這些類別濾除，訓練網頁用的模型時再用上就好


## 檔案用途說明

### [`labelme2coco.py`](labelme2coco.py)
指定需要轉換的原始資料路徑、輸出資料夾位置名稱，輸出物件數量，即可將labelme格式轉為coco資料集格式，`check_coco.py`, `pycocoDemo.ipynb`: 都是確認資料集正確性的工具


### [`convert_labelme_to1920.py`](convert_labelme_to1920.py)、[`convert_folder_imgs_size.py`](convert_folder_imgs_size.py)
將原始較高的圖片解析度降低為最長邊1920，標記與圖片一併修改，以減低模型evaluate難度，因輸入模型都還會再縮減尺吋(default=1333)


### [`compress_dataset.bat`](compress_dataset.bat)
windows bat檔案，將原始標記json檔案排除，只壓縮照片以及coco總label檔案，裡面的資料夾與輸出位置可以用文字編輯器編輯


### [`analyze_coco_dataset.py`](analyze_coco_dataset.py)
分析製作好的COCO資料集中，train/val照片與物件數量，計算相對應比例


### [`extract_ValSet_data.py`](extract_ValSet_data.py)
提取出對應COCO資料集中的val照片到指定資料夾中，以便模型進行測試

```bash
python3 extract_ValSet_data.py \
    --dataset_Name "20231213_ValidationSet_0point1" \
    --coco_json_folder "COCO_Format" \
    --img_source_folder "./" \
    --extract_type "json"
```


## Dataset資料夾架構與內容說明:

```
Adam_pseudo_label: 。世鈺進行的pseudo label標註，共705張
	|__ 202111_patrol: (164張)。11月農民遙控載具拍攝，
	|__ justin_remain: (541張)。熊哥時期拍攝影像，從2021 11月影像挑出來標記，感覺應該有篩選過了

COCO_Format: 依據資料夾存放各個時期的coco annotation.json，不儲存原始影像與標記檔案

Justin_labeled_data: (1026張)。熊哥標註的檔案

Justin_unlabeled_data: (1077張)。熊哥未標註照片，都被挑選過了，剩下的可能是沒那麼代表性的照片，優先度低

robot_regular_patrol: 農民遙控車輛定期拍攝的結果
	|__ 202111_unlabel: (316張)。已經都被挑選過了，剩下的可能是沒那麼代表性的照片，優先度低
	|__ 20210922_30: (241張)。懷樂同學標記
	|__ 2021102,03,22...30: (480張)。廷睿煒翔協助世鈺標記
```


註: 目前處理的檔案都是原始拍攝照片，沒有加入copy-paste硬寫進去的結果