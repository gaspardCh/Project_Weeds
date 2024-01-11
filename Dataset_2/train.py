from ultralytics import YOLO
import torch
import os
import random

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model.to('cuda')
# Use the model
# model.train(data="config.yaml", epochs=15)  # train the model

image_dir = r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main" \
            r"\cropandweed-dataset-main\data\images"
image_list = os.listdir(image_dir)
model = YOLO(r"C:\Users\gaspa\Documents\ENSEA\SIA\ProjetSIA_Chapron\ImageRecognition-main\cropandweed-dataset-main"
             r"\cropandweed-dataset-main\runs\detect\train3\weights\best.pt")
n = 3

for k in range(n):
    image = random.choice(image_list)
    model.predict(
        source=os.path.join(image_dir, image),
        save=True, show=True)


