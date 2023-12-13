"""
使用前把所有json檔案放到下一層資料夾中，只需要留下json檔案就好
程式會統計各個檔案內的母莖數量，並統計成csv檔案
"""

import os
import json
import csv
from tqdm import tqdm


# 存储统计结果的列表
results = []
directory = r"D:\NTU\研究相關\Dataset\蘆筍dataset\訓練用\第一次訓練資料集\train_copy"

def custom_sort(filename):
    # 检查文件类型
    if filename.endswith('.json'):
        return (1, int(filename[:-5]))
    elif filename.endswith('.jpg'):
        return (2, int(filename[:-4]))
    else:
        return (3, filename)

# 遍历所有的json文件
for filename in tqdm(sorted(os.listdir(directory), key=custom_sort)):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r') as f:
            data = json.load(f)

        # 统计stalk与spear类别的数量
        stalk_count = 0
        spear_count = 0
        for shape in data['shapes']:
            if shape['label'] == 'stalk':
                stalk_count += 1
            elif shape['label'] == 'spear':
                spear_count += 1

        # 将文件名和stalk数量加入列表
        results.append([filename[:-5], stalk_count, spear_count])

# 将结果写入csv文件
with open('stalk_spear_count_copy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'stalk count', 'spear count'])
    writer.writerows(results)
