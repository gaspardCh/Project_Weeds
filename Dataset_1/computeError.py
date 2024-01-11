import os
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Le chemin d'accès au dossier
folder_path = "dataset/test/"

IMG_HEIGHT = 512
IMG_WIDTH = 512

# Charger le modèle sauvegardé
model = keras.models.load_model('model.h5')
actual = []
predicted = []
# Parcourir chaque fichier du dossier
for filename in os.listdir(folder_path):
    # Le chemin complet du fichier
    file_path = os.path.join(folder_path, filename)

    # Charger l'image que vous souhaitez prédire
    image = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Convertir l'image en tableau numpy
    image_array = img_to_array(image)

    # Étendre les dimensions du tableau numpy pour qu'il ait la forme (1, 224, 224, 3)
    image_array = image_array.reshape((1,) + image_array.shape)

    # Prétraiter les données
    image_array = image_array / 255.0
    image_name = filename[0:3]
    image_class = 0 if image_name == "coc" else (1 if image_name == "fox" else (2 if image_name == "pig" else 3))
    # Faire une prédiction sur l'image
    prediction = model.predict(image_array)
    index_max_prediction = np.argmax(prediction)
    class_prediction = "Cocklebur" if index_max_prediction == 0 else (
        "Foxtail" if index_max_prediction == 1 else ("Pigweed" if index_max_prediction == 2 else "Ragweed"))
    actual.append(image_class)
    predicted.append(index_max_prediction)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Cocklebur", "Foxtail", "Pigweed", "Ragweed"])

cm_display.plot()
plt.show()
