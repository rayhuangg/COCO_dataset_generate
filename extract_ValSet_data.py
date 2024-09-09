import shutil
import argparse
from pathlib import Path
from pycocotools.coco import COCO

def copy_coco_val_data(dataset_name, COCO_source_folder, IMG_source_folder, destination_folder, extract_type):
    # Load COCO validation set annotations
    val_annotation_path = Path(COCO_source_folder) / dataset_name / 'instances_val2017.json'
    coco_val = COCO(val_annotation_path)

    # Create destination folder if it doesn't exist
    destination_folder_path = Path(destination_folder)
    destination_folder_path.mkdir(parents=True, exist_ok=True)

    # Get image IDs in the validation set
    val_image_ids = coco_val.getImgIds()

    # Copy each image and its annotation file to the destination folder
    for img_id in val_image_ids:
        img_info = coco_val.loadImgs(img_id)[0]

        # Copy image file
        if "photo" in extract_type:
            source_img_path = Path(IMG_source_folder) / img_info["file_name"]
            destination_img_path = destination_folder_path / Path(img_info["file_name"]).name
            shutil.copyfile(source_img_path, destination_img_path)

        # Copy annotation file
        if "json" in extract_type:
            source_ann_path = Path(IMG_source_folder) / Path(img_info["file_name"]).with_suffix('.json')
            destination_ann_path = destination_folder_path / Path(img_info["file_name"]).with_suffix('.json').name
            shutil.copyfile(source_ann_path, destination_ann_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', '--dataset_Name', type=str, default="20231213_ValidationSet_0point1", help='Name of the dataset folder to be analyzed')
    parser.add_argument('--coco_json_folder', type=str, default='COCO_Format', help='Path to COCO json file source folder')
    parser.add_argument('--img_source_folder', type=str, default="./", help='Path to image source folder')
    parser.add_argument('--extract_type', nargs='+', default="photo", help='Types of data to extract, photo or json')

    args = parser.parse_args()

    destination_folder = f'val_set_extract/{args.dataset_Name}'
    copy_coco_val_data(args.dataset_Name, args.coco_json_folder, args.img_source_folder, destination_folder, args.extract_type)