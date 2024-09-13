import cv2
import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from os import listdir
from scipy.signal import convolve
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

# Função para carregar imagens e rótulos para o treinamento do LBPH
def generateData(people):
    face_data = []
    labels = []
    for idx, person in enumerate(people):
        for picture in person["pictures"]:
            image = cv2.imread(f"{PATH}/processed/{picture}")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_image[y: y + h, x: x + w]
                face_roi = resize_image(face_roi)
                face_data.append(face_roi)
                labels.append(idx)  # Usa o índice da pessoa como rótulo

    return face_data, labels

# Geração do dataset e treino do LBPH Face Recognizer
def generateDataset(people):
    face_data, labels = generateData(people)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Treina o LBPH com as faces e os rótulos
    recognizer.train(face_data, np.array(labels))

    # Salva o modelo treinado
    recognizer.save("lbph_face_model.yml")

    print("[LBPH] Treinamento concluído e modelo salvo!")

    return recognizer

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
    recognizer = generateDataset(people)

    return recognizer, people

def main():
    recognizer, people = setup()

    video = cv2.VideoCapture(0)

    if video.isOpened() == False:
        print("> CÂMERA NÃO ESTÁ ABERTA!")
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

            # Reconhecimento usando LBPH
            label, confidence = recognizer.predict(face_roi)

            # Usa o índice do rótulo para obter o nome da pessoa
            name = NAMES[label]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de face", frame)

        key = cv2.waitKey(30)

        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
