# Importar libreria cv2
# Instalar cv2  pip3 install opencv-python
import cv2
import os
import json



# Función para guardar una imagen en la carpeta img
def save_image(ruta, text, img):
    # Usar os.path.splitext() para separar el nombre y la extensión
    nombre, extension = os.path.splitext(ruta)
    # Crear la nueva ruta con el nombre modificado y la misma extensión
    nueva_imagen = nombre + text + extension
    # La guardamos en memoria en un fichero distinto y devolvemos la ruta
    cv2.imwrite(nueva_imagen, img)


###########################################################################################
# Caso 1
# Se desea diseñar una aplicación que aplique difuminado a los rostros de una imagen dada,
# generando una nueva imagen, a través de la información extraída de un json usando AWS Rekognition.
###########################################################################################

def blur_faces(ruta_img,json_path):

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

    save_image(ruta_img,'_Dif',img)

###########################################################################################
# Caso 2
# A partir del caso práctico anterior, se desea diseñar una aplicación similar pero que solo se le
# aplique difuminado a los rostros que se identifiquen como menores, para ello se atenderá a la edad
# mínima en la horquilla de la clasificación.
###########################################################################################

def blur_menor(ruta_img,json_path):
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
        age = face.get('AgeRange',{})
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