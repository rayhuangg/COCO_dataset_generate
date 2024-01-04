Modify time: 20230724
---
目前處理的檔案都是原始拍攝照片，沒有加入copy-paste硬寫進去的結果

## 轉檔使用方法
`labelme2coco.py`: 指定需要轉換的原始資料路徑、輸出資料夾位置名稱，輸出物件數量，即可將labelme格式轉為coco資料集格式
`check_coco.py`, `pycocoDemo.ipynb`: 都是確認資料集正確性的工具
`convert_labelme_to1920.py`: 將原始較高的圖片解析度降低為最長邊1920，標記與圖片一併修改，以減低模型evaluate難度，因輸入模型都還會再縮減尺吋(default=1333)
`compress_dataset.bat`: windows bat檔案，將原始標記json檔案排除，只壓縮照片以及coco總label檔案，裡面的資料夾與輸出位置可以用文字編輯器編輯
`analyze_coco_dataset.py`: 分析製作好的COCO資料集中，train/val照片與物件數量，計算相對應比例
`extract_ValSet_data.py`: 提取出對應COCO資料集中的val照片到指定資料夾中，以便模型進行測試

---
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

---
## 照片編號說明:

熊哥原始拍攝資料1~2650
	熊哥標記1~388，1986~2588、2616~2650
	世鈺pseudo label標記390~1982(有跳著選)、2589~2614
	總共未標記1077張(已確認)

	照號碼排序結果:
	1~388: 熊哥標記
	390~1982: 世鈺pseudo label標記
	1986~2588: 熊哥標記
	2589~2614: 世鈺pseudo label標記(有跳著選)
	2616~2650: 熊哥標記


懷樂同學標註資料: 3000~3240，202109月拍攝
廷睿煒翔標註資料: 原始為4000~4239.jpg，有找到原始資料(202111月拍攝)，已改掉，原始位置:(蘆筍dataset\adam\old_data")