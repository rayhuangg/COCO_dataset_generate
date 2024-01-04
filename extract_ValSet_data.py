from pathlib import Path
import shutil
from pycocotools.coco import COCO

def copy_coco_val_data(COCO_source_folder, IMG_source_folder, destination_folder, dataset_name, extract_type):
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
    dataset_name = "20231213_ValidationSet_0point1"
    COCO_source_folder = 'COCO_Format'
    IMG_source_folder = "./"
    destination_folder = f'val_set_extract/{dataset_name}'
    extract_type = ["photo"] # or ["photo", "json"]

    copy_coco_val_data(COCO_source_folder, IMG_source_folder, destination_folder, dataset_name, extract_type)
