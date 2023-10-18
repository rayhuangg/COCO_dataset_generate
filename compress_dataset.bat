REM 將下列資料夾全部壓縮在同一zip檔案內，並排除json標註檔案以減少體積
zip -r Asparagus_Dataset.zip Adam_pseudo_label -x "*.json"
zip -r Asparagus_Dataset.zip Justin_labeled_data -x "*.json"
zip -r Asparagus_Dataset.zip Justin_labeled_data_1920 -x "*.json"
zip -r Asparagus_Dataset.zip robot_regular_patrol -x "*.json"

REM 壓縮COCO資料夾 並且保留json總標記
zip -r Asparagus_Dataset.zip COCO_Format
