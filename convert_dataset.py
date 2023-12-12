from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import csv
from tqdm import tqdm
import os

# Init coco object
bbox_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main\cropandweed-dataset-main\data\bboxes"
images_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main\cropandweed-dataset-main\data\images"
image_width = 1920
image_height = 1088

# Get list of classes : eval_list_dir contain all directory with bounding box
full_list_dir = os.listdir(bbox_dir)
eval_list_dir = []
for directory in full_list_dir:
    if "Eval" in directory:
        eval_list_dir.append(directory)

# Get list of classes : list_category contain the 3 first letter of each class
list_category = []
for eval_dir in eval_list_dir:
    category = eval_dir[0:3]
    if category not in list_category:
        list_category.append(category)

# Coco_image_list contains list of coco image object
image_list = os.listdir(images_dir)
coco_image_list = []
for k, image in enumerate(image_list):
    coco_image = CocoImage(file_name=os.path.join(images_dir, image), height=1088, width=1920)
    coco_image_list.append(coco_image)

# Create the model and add classes from list_category
coco = Coco()
for k, category in enumerate(list_category):
    coco.add_category(CocoCategory(id=k, name=category))

# For each directory
for eval_dir in tqdm(eval_list_dir):
    category = list_category.index(eval_dir[0:3])
    list_file = os.listdir(os.path.join(bbox_dir, eval_dir))

    if len(list_file) != len(image_list):
        raise Exception("not the same number of image in images and " + eval_dir)

    for i, file in enumerate(tqdm(list_file)):
        path = os.path.join(bbox_dir, eval_dir, file)
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                bbox_coord = row[0].split(',')
                x1 = int(bbox_coord[0])
                y1 = int(bbox_coord[1])
                x2 = int(bbox_coord[2])
                y2 = int(bbox_coord[3])
                x_center = int(bbox_coord[4])
                y_center = int(bbox_coord[5])
                coco_image_list[i].add_annotation(
                    CocoAnnotation(
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        category_id=category,
                        category_name=list_category[category]
                    )
                )
for img in coco_image_list:
    coco.add_image(img)

save_json(data=coco.json, save_path="cocoDataset.json")
