import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Definimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)
        
        
def PreProcess (Url):
    """Abre la imagen, la pasa a escala de grises y la Binariza con un Threshold Gaussiano. Devuelve Imagen Binarizada"""
    Img = cv2.imread(Url) 
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # imshow(Img, title="Imagen Original")

    gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    # imshow(gray, title="Imagen en escala de grises")



    Binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 13)
    # imshow(Binary)
    return Img , Binary

def Contours(Binary):
    """Funcion que se encarga de encontrar y dibujar todos los contornos en una mascara nueva."""
    contornos = cv2.findContours(Binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )[0]
    canvas = np.zeros_like(Binary)
    cv2.drawContours(canvas , contornos, -1, (0, 255, 0), 1)
    # imshow(canvas)
    return contornos

def filter_candidates(contours):
    
    """Esta funcion la utilizamos para 'Filtrar' los candidatos. Y hablamos de los candidatos como todos los 'contornos' que dibujos openCv en la funcion anterior.
    Utilizaremos 2 criterios, un aspect ratio que es un coeficiente que sacamos de dividir el alto por el ancho de una patente 'vieja' reglamentaria, y asi obtener ese factor, 
    y el area para eliminar pequeños ruidos."""
    candidates = []
    # 29,4 cm. por 12,9 cm = 294 mm y 129 mm => Factor de patente = 294 / 129 = 2.27906977
    ratio = 2.27906977
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        if np.isclose(aspect_ratio, ratio, atol=0.7) and area > 1000:
            candidates.append(cnt)
    return candidates


def eleccion_mejor_candidato(candidates, Img):
    """Esta funcion lo que hace es de los candidatos pre-filtrados anteriormente, elige el de mas abajo, es una burda manera de quedarnos con la patente, ya que siempre estan en la parte inferior de los autos."""
    canvas = np.zeros_like(Img)
    cv2.drawContours(canvas , candidates, -1, (0, 255, 0), 1)

    ys = []
    for cnt in candidates:
        x, y, w, h = cv2.boundingRect(cnt)
        ys.append(y)

    license = candidates[np.argmax(ys)]
    canvas = np.zeros_like(Img)
    cv2.drawContours(canvas , [license], -1, (0, 255, 0),  thickness=cv2.FILLED)

    return license
 
def obtencion_patente(mejor_candidato, Img):
    """Esta funcion lo que hace es obtener la patente en el lugar previamente seleccionado y a su vez crear una mascara con solo esa informacion.
    Retorna una imagen en negro, solo con informacion donde estaba el 'mejor candidato' a ser patente."""
    x, y, w, h = cv2.boundingRect(mejor_candidato)
    cropped = Img[y:y+h, x:x+w]
    # imshow(cropped)

    #Creo una nueva imagen con la patente
    new_img = np.zeros_like(Img)


    h, w = cropped.shape[:2]


    y_start, y_end = y, y + h  
    x_start, x_end = x, x + w 


    new_img[y_start:y_end, x_start:x_end] = cropped
    
    coordenas = [x,y,w,h]
    
    return new_img , coordenas

def procesamiento_patente(patente):
    """Esta funcion lo que hara es binarizar, umbralizar y detectar las componentes conectadas de la patente en cuestion, con el fin de poder analizar su contenido (caracteres).
    Como resultado dara 2 imagenes, una con los caracteres limpios detectados y otra con los mismos con sus respectivos Bounding boxes."""
    #La convierto en escala de grises y la umbralo
    new_img = cv2.cvtColor(patente, cv2.COLOR_RGB2GRAY)
    th , new_img = cv2.threshold(new_img, 120, 255, cv2.THRESH_BINARY)

    #Le hago las componentes conectadas
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_img, connectivity, cv2.CV_32S)  
    # imshow(labels, title="Componentes Conectados")

    #Detecto los caracteres
    Caracteres_Img_Original = np.zeros_like(new_img)
    BB_Caracter = np.zeros_like(new_img)
    for jj in range(1, num_labels):
        altura = stats[jj, cv2.CC_STAT_HEIGHT]
        ancho = stats[jj, cv2.CC_STAT_WIDTH]
        area = stats[jj, cv2.CC_STAT_AREA]
        rel_asp = stats[jj, cv2.CC_STAT_HEIGHT] / stats[jj, cv2.CC_STAT_WIDTH]
        
        
        if (rel_asp<1.5) or (rel_asp >3.0) or np.isclose(altura, 13, rtol= 2)  :

            Caracteres_Img_Original[labels == jj] = new_img[labels == jj]
            BB_Caracter[labels == jj] = new_img[labels == jj]
            start_point = (stats[jj, 0], stats[jj, 1]) 
            end_point = (stats[jj, 0] + stats[jj, 2], stats[jj, 1] + stats[jj, 3]) 
            cv2.rectangle(BB_Caracter, start_point, end_point, (255, 0, 0), 1)
    
    
    return Caracteres_Img_Original , BB_Caracter

def caracteres_clean(Caracteres_Img_Original, coordenadas):
    """Esta funcion lo que hace es en base a los caracteres detectados, les hace zoom y los muestra limpios y claros."""
    
    x,y,w,h = coordenadas
    cropped_caracters = Caracteres_Img_Original[y:y+h, x:x+w]
    
    
    return cropped_caracters


def imagen_final(Img, patente, Caracteres_Img_Original, BB_Caracter, cropped_caracters ,url):
    """Esta funcion lo que hara es visualizar el resultado del algoritmo y a su vez descargar la imagen final"""
    plt.figure(figsize=(10, 10))

    plt.subplot(321)
    plt.imshow(Img)
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(patente)
    plt.title("Patente")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(Caracteres_Img_Original, cmap="gray")
    plt.title("Deteccion de caracteres de la patente.")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(BB_Caracter, cmap="gray")
    plt.title("Bounding Box")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(cropped_caracters, cmap="gray")
    plt.title("Caracteres Clean")
    plt.axis('off')

    plt.tight_layout()

    # archivo = url.split("/")[-1]
    # nombre = f"Resultado_{archivo}"
    # plt.savefig(nombre, dpi=300)

    plt.show(block=False)


def main():
    """Programa principal. Obtiene todas las URLS de la carpeta patentes y ejecuta el resto de funciones"""
    # Ruta de la carpeta que contiene las imágenes
    carpeta_patentes = "patentes"

    # Obtener la lista de archivos en la carpeta
    archivos_en_carpeta = os.listdir(carpeta_patentes)
    archivos_en_carpeta
    for nombre_archivo in archivos_en_carpeta:
        # Construir la ruta completa del archivo
        url = f"{carpeta_patentes}/{nombre_archivo}"
        url
        Img,Binary  = PreProcess(url)
        contornos = Contours(Binary)
        # imshow(contornos)
        candidates = filter_candidates(contornos)
        # imshow(candidates)
        mejor_candidato = eleccion_mejor_candidato (candidates , Img)
        # imshow(mejor_candidato)
        patente, coordenadas = obtencion_patente(mejor_candidato, Img)
        # imshow(patente)
        Caracteres_Img_Original , BB_Caracter = procesamiento_patente(patente)
        # imshow(Caracteres_Img_Original)
        # imshow(BB_Caracter)
        cropped_caracters = caracteres_clean(Caracteres_Img_Original, coordenadas)
        # imshow(cropped_caracters)
        imagen_final(Img, patente, Caracteres_Img_Original, BB_Caracter, cropped_caracters ,url)

main()



