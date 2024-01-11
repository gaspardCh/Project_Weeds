from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import csv
from tqdm import tqdm
import os

# Directory of bounding boxes
bbox_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main\cropandweed-dataset-main\data\bboxes\CropOrWeed2"
# Directory of images
images_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main\cropandweed-dataset-main\data\images"
# Directory to save labels as txt file
dir_path = r'C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main\cropandweed-dataset-main\data\labels'

image_width = 1920
image_height = 1088

image_list = os.listdir(images_dir)
bbox_list = os.listdir(bbox_dir)
for image in tqdm(image_list):

    name_txt = dir_path + '/' + image
    name_csv = bbox_dir + '/' + image
    f = open(name_txt[:-4] + '.txt', 'w+')
    csv_file = name_csv[:-4] + '.csv'
    if os.path.basename(csv_file) in bbox_list:
        path = os.path.join(bbox_dir, csv_file)
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                bbox_coord = row[0].split(',')
                x1 = float(bbox_coord[0])/image_width
                y1 = float(bbox_coord[1])/image_height
                x2 = float(bbox_coord[2])/image_width
                y2 = float(bbox_coord[3])/image_height
                id = int(bbox_coord[4])
                x_center = float(bbox_coord[5])/image_width
                y_center = float(bbox_coord[6])/image_height
                width = x2 - x1
                height = y2 - y1

                f.write(str(id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
                f.write("\n")
