import cv2

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from os import listdir
from skimage.feature import hog
from scipy.signal import convolve

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DATASET = "dataset.csv"
PATH = "C:\\Users\\gabri\\OneDrive\\Documentos\\UFSCar\\PDI\\Reconhecimento\\eface-recognition\\img"

face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

filtro_lap = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

NAMES = [
    "João Pedro Silva",
    "Lucas Fernandes Almeida",
    "Gabriel Souza Oliveira",
    "Matheus Rocha Santos",
    "Felipe Costa Pereira",
    "Rafael Lima Duarte",
    "Bruno Araújo Ribeiro",
    "Gustavo Martins Carvalho",
    "Thiago Mendes Barbosa",
    "Leonardo Fonseca Andrade",
    "Ana Carolina Freitas",
    "Diego Moraes Batista",
    "Rodrigo Nascimento Cunha",
    "Daniel Alves Teixeira",
    "André Cardoso Ramos"]

def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def processImages():
    images = listdir(f"{PATH}/original")

    for index in range(0, len(images)):
        pixels = plt.imread(f"{PATH}/original/{images[index]}")
        pixels_float = pixels.astype(float)
        pixels_smooth = ndi.gaussian_filter(pixels_float, sigma=0)
        pixels_lap = convolve(pixels_smooth, filtro_lap, 'same')
        pixels_novo = pixels_smooth + 10 * pixels_lap

        images[index] = f"{images[index]}.png"
        plt.imsave(f"{PATH}/processed/{images[index]}", pixels_novo, cmap='gray')

    return images


def generateData(people):
    for _, person in enumerate(people):
        for picture in person["pictures"]:
            image = cv2.imread(f"{PATH}/processed/{picture}")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_image[y: y + h, x: x + w]
                face_roi = resize_image(face_roi)
                features = extract_hog_features(face_roi)
                
                person["data"].append(features)

def generateDataset(people):
    features = []
    pictures = []
    names = []

    for _, person in enumerate(people):
        for index in range(0, len(person["data"])):
            pictures.append(person["pictures"][index])
            features.append(person["data"][index])
            names.append(person["name"])

    dataframe = pd.DataFrame(features)

    dataframe["picture"] = pictures
    dataframe["name"] = names

    dataframe.to_csv(DATASET)

def setup():
    print("[APLICANDO FILTRO NAS IMAGENS]")
    images = processImages()

    rows = 15

    people = []
        
    for index in range(1, rows + 1):
        pictures = [image for image in images if f"subject{index:02d}." in image]

        people.append({
            "name": NAMES[index - 1],
            "pictures": pictures,
            "data": []
        })

    print("[EXTRAINDO CARACTERÍSTICAS DO ROSTO]")
    generateData(people)

    print("[GERAR DATASET]")
    generateDataset(people)

def training():
    print("[TREINAMENTO]")
    dataset = pd.read_csv("dataset.csv")

    X = dataset.drop(columns=["name", "picture", dataset.columns[0]])
    y = dataset["name"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    X_output = dataset.loc[X_test.index, ["picture", "name"]]

    for index in range(0, len(X_output)):
        image = cv2.imread(f"{PATH}/processed/{X_output["picture"].iloc[index]}")

        cv2.putText(image, X_output["name"].iloc[index], (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f"{PATH}/test/{X_output["picture"].iloc[index]}", image)

    ctf = KNeighborsClassifier(n_neighbors=1)
    ctf.fit(X_train, y_train)

    print(f"KNN Acurácia: {ctf.score(X_test, y_test)}")

    return ctf

def main():
    setup()

    ctf = training()

    video = cv2.VideoCapture()

    video.open(0)

    if video.isOpened() == False:
        print("> CÂMERA NÂO ESTÁ ABERTA!")
        exit(0)

    while True:
        success, frame = video.read()

        if success == False:
            print("> NÃO FOI POSSÍVEL OBTER IMAGEM DA CÂMERA!")
            exit(0)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_image[y: y + h, x: x + w]
            face_roi = resize_image(face_roi)
            features = extract_hog_features(face_roi)

            name = ctf.predict([features])[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Reconhecimento de face", frame)

        key = cv2.waitKey(30)

        if key == 27:
            break


if __name__ == "__main__":
    main()