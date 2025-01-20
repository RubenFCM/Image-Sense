# Importar libreria cv2
# Instalar cv2  pip3 install opencv-python
from email.policy import default

import cv2
import os
import json


# Función para guardar una imagen en la carpeta img
def save_image(img,ruta_salida, nombre_imagen, ruta_img,default):
    # Si no se proporciona una ruta de salida, usa una por defecto
    if not ruta_salida:
        ruta_salida = '../imagenes/'

    # Si no se especifica un nombre de imagen, usar el nombre de la imagen original
    if not nombre_imagen:
        nombre_imagen = os.path.splitext(os.path.basename(ruta_img))[0] + default

    # Asegúrate de que ruta_salida tenga un nombre de archivo válido
    if not os.path.splitext(ruta_salida)[1]:  # Si no tiene extensión
        ruta_salida = os.path.join(ruta_salida, nombre_imagen + '.jpg')

    # Asegurarse de que la ruta de salida sea válida
    if not os.path.exists(os.path.dirname(ruta_salida)) and os.path.dirname(ruta_salida):
        os.makedirs(os.path.dirname(ruta_salida))
    # Guardar la imagen con los cambios
    cv2.imwrite(ruta_salida, img)


###########################################################################################
# Caso 1
# Se desea diseñar una aplicación que aplique difuminado a los rostros de una imagen dada,
# generando una nueva imagen, a través de la información extraída de un json usando AWS Rekognition.
###########################################################################################

def blur_faces(ruta_img, json_path, nombre_imagen = '',ruta_salida=''):
    img = cv2.imread(ruta_img)
    # Obtener las dimensiones de la imagen
    image_height, image_width, _ = img.shape

    with open(json_path, "r") as file:
        data = json.load(file)

    # Verificar si existen detalles de caras
    if "FaceDetails" not in data or not data["FaceDetails"]:
        raise ValueError("El archivo JSON no contiene información de las caras.")

    # Procesar cada cara detectada en el JSON
    for face in data["FaceDetails"]:
        bounding_box = face.get("BoundingBox", {})
        if not bounding_box:
            continue

        # Calcular las coordenadas absolutas de la cara
        x = int(bounding_box["Left"] * image_width)
        y = int(bounding_box["Top"] * image_height)
        width = int(bounding_box["Width"] * image_width)
        height = int(bounding_box["Height"] * image_height)

        # Recortar el área de la cara
        face_region = img[y:y + height, x:x + width]

        # Aplicar un filtro Gaussiano para difuminar la región
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)

        # Reemplazar la región original con la difuminada
        img[y:y + height, x:x + width] = blurred_face

    save_image(img,ruta_salida, nombre_imagen, ruta_img,'_Dif')



###########################################################################################
# Caso 2
# A partir del caso práctico anterior, se desea diseñar una aplicación similar pero que solo se le
# aplique difuminado a los rostros que se identifiquen como menores, para ello se atenderá a la edad
# mínima en la horquilla de la clasificación.
###########################################################################################

def blur_menor(ruta_img, json_path):
    img = cv2.imread(ruta_img)
    # Obtener las dimensiones de la imagen
    image_height, image_width, _ = img.shape

    with open(json_path, "r") as file:
        data = json.load(file)

    # Verificar si existen detalles de caras
    if "FaceDetails" not in data or not data["FaceDetails"]:
        raise ValueError("El archivo JSON no contiene información de las caras.")

    # Procesar cada cara detectada en el JSON
    for face in data["FaceDetails"]:
        bounding_box = face.get("BoundingBox", {})
        age = face.get('AgeRange', {})
        if not bounding_box and not age:
            continue

        if age['Low'] < 18:
            # Calcular las coordenadas absolutas de la cara
            x = int(bounding_box["Left"] * image_width)
            y = int(bounding_box["Top"] * image_height)
            width = int(bounding_box["Width"] * image_width)
            height = int(bounding_box["Height"] * image_height)

            # Recortar el área de la cara
            face_region = img[y:y + height, x:x + width]

            # Aplicar un filtro Gaussiano para difuminar la región
            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)

            # Reemplazar la región original con la difuminada
            img[y:y + height, x:x + width] = blurred_face

    save_image(ruta_img, '_DifMen', img)


###########################################################################################
# Caso 3
# A partir de los casos prácticos anteriores, se desea diseñar una aplicación que realice
# reconocimiento facial y clasificación de rostros. Se marcarán los rostros con un marco y una
# etiqueta ajustada al mismo.

# Si el rostro se corresponde con un menor, será de color amarillo
# Si el rostro se corresponde con un hombre, será de color rojo.
# Si el rostro se corresponde con una mujer, será de color verde.

###########################################################################################

def square_faces(ruta_img, json_path):
    img = cv2.imread(ruta_img)
    # Obtener las dimensiones de la imagen
    image_height, image_width, _ = img.shape

    with open(json_path, "r") as file:
        data = json.load(file)

    # Verificar si existen detalles de caras
    if "FaceDetails" not in data or not data["FaceDetails"]:
        raise ValueError("El archivo JSON no contiene información de las caras.")

    # Procesar cada cara detectada en el JSON
    for face in data["FaceDetails"]:
        bounding_box = face.get("BoundingBox", {})
        age = face.get('AgeRange', {})
        gender = face.get('Gender',{})
        if not bounding_box and not age:
            continue
        # Calcular las coordenadas absolutas de la cara
        x = int(bounding_box["Left"] * image_width)
        y = int(bounding_box["Top"] * image_height)
        width = int(bounding_box["Width"] * image_width)
        height = int(bounding_box["Height"] * image_height)

        if gender['Value'] == 'Male' and age['Low'] >= 18:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
        elif gender['Value'] == 'Female' and age['Low'] >= 18:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 255), 2)

    save_image(ruta_img, '_Box', img)

###########################################################################################
# Caso 4
# En la línea de los casos prácticos anteriores, se desea aplicar reconocimiento facial a una imagen,
# reconociendo los distintos rostros que aparecen y permitiendo el etiquetado de los mismos.

# Se debe articular un sistema que permita al usuario indicar el nombre (o una etiqueta) a un rostro,
# dichos datos se almacenarán en un documento xml/json que contendrá toda la información relativa
# al proceso realizado, la imagen original, los rostros detectados (posición, nombre, edad, sexo,
# estado de ánimo, etc).

# Además se diseñará una función que permita aplicar un xml generado anteriormente a una imagen
# con el objetivo de generar una nueva imagen donde se hacen visibles los datos que contiene.

###########################################################################################