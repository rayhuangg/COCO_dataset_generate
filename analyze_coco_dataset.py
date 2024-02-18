from pathlib import Path
from pycocotools.coco import COCO

def main():
    # Load the annotation files for the training and validation sets
    dataset_name = "20230817_Adam_1920"
    train_annotation_path = Path('COCO_Format') / dataset_name / 'instances_train2017.json'
    val_annotation_path = Path('COCO_Format') / dataset_name / 'instances_val2017.json'

    # Initialize COCO objects
    coco_train = COCO(train_annotation_path)
    coco_val = COCO(val_annotation_path)

    # Get the number of images in the training and validation sets
    num_train_images = len(coco_train.getImgIds())
    num_val_images = len(coco_val.getImgIds())

    # Get the category IDs for the training and validation sets
    train_cat_ids = coco_train.getCatIds()
    val_cat_ids = coco_val.getCatIds()

    # Get the number of instances for each category in the training and validation sets
    num_instances_train = [len(coco_train.getAnnIds(catIds=cat_id)) for cat_id in train_cat_ids]
    num_instances_val = [len(coco_val.getAnnIds(catIds=cat_id)) for cat_id in val_cat_ids]

    # Find the indices of 'stalk' and 'spear' in the category IDs
    stalk_index_train = train_cat_ids.index(coco_train.getCatIds(catNms=['stalk'])[0])
    spear_index_train = train_cat_ids.index(coco_train.getCatIds(catNms=['spear'])[0])

    stalk_index_val = val_cat_ids.index(coco_val.getCatIds(catNms=['stalk'])[0])
    spear_index_val = val_cat_ids.index(coco_val.getCatIds(catNms=['spear'])[0])

    # Get the number of instances for 'stalk' and 'spear' in the training and validation sets
    num_instances_stalk_train = num_instances_train[stalk_index_train]
    num_instances_spear_train = num_instances_train[spear_index_train]

    num_instances_stalk_val = num_instances_val[stalk_index_val]
    num_instances_spear_val = num_instances_val[spear_index_val]

    # Calculate ratios
    image_ratio = num_train_images / num_val_images
    ratio_stalk_train = num_instances_stalk_train / num_instances_spear_train if num_instances_spear_train != 0 else float('inf')
    ratio_stalk_val = num_instances_stalk_val / num_instances_spear_val if num_instances_spear_val != 0 else float('inf')

    # Display the results
    print("\nResults Summary:")
    print(f"  Number of images in the training set: {num_train_images}")
    print(f"  Number of images in the validation set: {num_val_images}")
    print(f"  Image ratio (training:validation): {image_ratio:0.2f}:1\n")

    # Training set instances
    print("Number of instances for each category in the TRAINING set:")
    for cat_id, num_instances in zip(train_cat_ids, num_instances_train):
        category_info = coco_train.loadCats(cat_id)[0]
        print(f"  {category_info['name']}: {num_instances}")

    # Validation set instances
    print("\nNumber of instances for each category in the VALIDATION set:")
    for cat_id, num_instances in zip(val_cat_ids, num_instances_val):
        category_info = coco_val.loadCats(cat_id)[0]
        print(f"  {category_info['name']}: {num_instances}")

    # Ratios
    print("\nRatios:")
    print(f"  Ratio of 'stalk' to 'spear' instances in the TRAINING set: {ratio_stalk_train:.2f}:1")
    print(f"  Ratio of 'stalk' to 'spear' instances in the VALIDATION set: {ratio_stalk_val:.2f}:1")



if __name__ == "__main__":
    main()