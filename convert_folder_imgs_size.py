from pathlib import Path
import cv2

def resize_photos(src_dir, dst_dir, target_width=1920):
    # 轉換成 Path 對象
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 檢查目標目錄是否存在，如果不存在則創建
    if not dst_path.exists():
        dst_path.mkdir(parents=True)

    # 使用 rglob 替代 os.walk，以包含所有子目錄
    for img_file in src_path.rglob('*.*'):
        # 處理照片
        process_photo(img_file, src_path, dst_path, target_width)

def process_photo(img_file, src_base, dst_base, target_width):
    file_relative_path = img_file.relative_to(src_base)
    file_dst_path = dst_base / file_relative_path

    # 檢查目標目錄是否存在，如果不存在則創建
    if not file_dst_path.parent.exists():
        file_dst_path.parent.mkdir(parents=True)

    file_name = img_file.name
    print(f"Processing {file_name}")

    # 讀取照片
    img = cv2.imread(str(img_file))

    # 檢查是否成功讀取照片
    if img is None:
        print(f"Failed to read {file_name}. Skipping.")
        return

    # 取得照片尺寸
    (h, w, c) = img.shape

    # 計算新的高度
    h_new = int(h / w * target_width)

    # 計算寬高比例
    ratio = target_width / w

    # 等比例縮放
    img_resize = cv2.resize(img, (target_width, h_new))

    # 寫入轉換後的照片到目標目錄
    cv2.imwrite(str(file_dst_path), img_resize)

# 執行程式碼
src_directory = r'C:\NTU\Asparagus_Dataset\purple'
dst_directory = r'C:\NTU\Asparagus_Dataset\purple_1920'
resize_photos(src_directory, dst_directory)
