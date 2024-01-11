import cv2
from ultralytics import YOLO
import os
import random

# directory of test images
image_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main" \
            r"\cropandweed-dataset-main\data\images"
image_list = os.listdir(image_dir)
# directory of anotations of test images
annotation_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main" \
                 r"\cropandweed-dataset-main\data\labels"
# directory of weights of the model
model = YOLO(r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main"
             r"\cropandweed-dataset-main\runs\detect\train3\weights\best.pt")
n = 5
# path to create image with bounding box to compare with prediction
path_prediction = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main" \
                  r"\cropandweed-dataset-main\runs\detect\predict"


def draw_bounding_boxes(image_path, annotation_path, output_path):
    # Charger l'image
    image = cv2.imread(image_path)

    # Lire les annotations YOLO depuis le fichier txt
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Les annotations YOLO sont au format (classe, centre_x, centre_y, largeur, hauteur)
        # Convertir ces valeurs en coordonnées de coin supérieur gauche et inférieur droit
        values = line.strip().split(' ')
        class_id, center_x, center_y, width, height = map(float, values)
        x, y, w, h = int((center_x - width / 2) * image.shape[1]), int((center_y - height / 2) * image.shape[0]), int(
            width * image.shape[1]), int(height * image.shape[0])

        # Dessiner la bounding box sur l'image
        color = (255, 255, 255)  # blanc (bgr)
        if class_id == 0:
            text = "Crop"
        else:
            text = "Weed"
        thickness = 4
        image1 = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        image = cv2.putText(
            img=image1,
            text=text,
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=3.0,
            color=color,
            thickness=5
        )

    # Enregistrer l'image avec les bounding boxes
    print(output_path)
    cv2.imwrite(output_path, image)


for k in range(n):
    image = random.choice(image_list)
    model.predict(
        source=os.path.join(image_dir, image),
        save=True)
    anot = image[:-4] + ".txt"
    real_image = image[:-4] + "real" + image[-4:]
    draw_bounding_boxes(os.path.join(image_dir, image), os.path.join(annotation_dir, anot),
                        os.path.join(path_prediction, real_image))
