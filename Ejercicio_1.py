import cv2
from matplotlib import pyplot as plt
import numpy as np

# --- Ejercicio 1 - Parte a)

# --- Cargo imagen --------------------------------------------------------------------------------
img = cv2.imread('monedas.jpg', cv2.IMREAD_COLOR)
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img_original, cmap='gray'), plt.show(block=False)
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(img_gris, cmap='gray'), plt.show(block=False)

# --- Suavizo --------------------------------------------------------------------------------------
img_fil = cv2.medianBlur(img_gris, 5)
plt.figure(), plt.imshow(img_fil, cmap='gray'), plt.show(block=False)

# --- Filtro pasabajo
img_pasabajo = cv2.blur(img_fil, (5, 5))
plt.figure(), plt.imshow(img_fil, cmap='gray'), plt.show(block=False)

# --- Detección de bordes con Canny
bordes = cv2.Canny(img_pasabajo, 10, 80)
plt.figure(), plt.imshow(bordes, cmap='gray'), plt.show(block=False)

# --- Cierro los circulos dilatando
kernel = np.ones((23,23), np.uint8)
dilatado = cv2.dilate(bordes, kernel, iterations=1)
plt.figure(), plt.imshow(dilatado, cmap='gray'), plt.show(block=False)

# --- Genero la máscara
num_labels, etiquetas= cv2.connectedComponents(dilatado)
mascara = np.zeros_like(dilatado)

# --- Dibujar cada objeto en la máscara
for label in range(1, num_labels):
    mascara[etiquetas == label] = 255
plt.figure(), plt.imshow(mascara, cmap='gray'), plt.show(block=False)

# --- Busco los contornos y los dibujo
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bordes_imagen = np.zeros_like(mascara)
cv2.drawContours(bordes_imagen, contornos, -1, 255, 1)
plt.figure(), plt.imshow(bordes_imagen, cmap='gray'), plt.show(block=False)

# --- Cuento los contornos 
len(contornos)

# --- Paso la tupla contornos a una lista y corroboro
contornos = list(contornos)
print(type(contornos))

# --- Claisifico por forma

img_final = img_original.copy()

# --- Consulto la excentricidad de todos los contornos 
for contorno in contornos:
        elipse = cv2.fitEllipse(contorno)
        excentricidad = elipse[1][0] / elipse[1][1]
        print(f"Excentricidad: {excentricidad}")

# Veo que la mayoría de los valores son superiores a 0.95 y hay algunos menores

# --- Genero las listas
monedas = []
dados = []
areas = []

# Itero sobre los contornos, busco la excentricidad y divido dos subgrupos, los circulos (excentricidad mayor a 0.95) y las otras figuras
k = -1
for contorno in contornos:
        k += 1
        elipse = cv2.fitEllipse(contorno)
        excentricidad = elipse[1][0] / elipse[1][1]
        if excentricidad > 0.95:
            # Agrego el contorno a la lista
            monedas.append(contorno)
            # Calculo el area y la agrego a la lista
            area = cv2.contourArea(contorno)
            areas.append(area)
            # Dibujo los contronos en la imagen y los diferencio con colores
            cv2.drawContours(img_final, contornos, k, (0,0,255),3)
        else:
            dados.append(contorno)
            cv2.drawContours(img_final, contornos, k, (0,255,0),3)
plt.figure(), plt.imshow(img_final), plt.show(block=False)

# --- Observo las listas
print(monedas)
print(dados)
print(areas)

# Veo que las areas son [101368.5, 101494.0, 71811.0, 71977.5, 76278.0, 72900.5, 116503.0, 69949.0, 75959.5, 100701.0, 115624.0, 69997.5, 99093.5, 103009.5, 75938.0, 72070.5, 119558.5]
areas.sort()
print(areas)
# Sabiendo que tenemos 3 grupos de monedas, tenemos que identificar 3 "saltos" en las areas
# Entonces vamos a dividir en 3 grupos segun los valores ordenados de las areas
# La lista ordenada queda: 
# [69949.0, 69997.5, 71811.0, 71977.5, 72070.5, 72900.5, 75938.0, 75959.5, 76278.0, 99093.5, 100701.0, 101368.5, 101494.0, 103009.5, 115624.0, 116503.0, 119558.5]
# chica <= 90000
# 90000 < mediana <= 1070000
# 107000 < grande 

# --- Inicializo las listas nuevamente
monedas = []
dados = []
monedas_10_centavos = 0
monedas_50_centavos = 0
monedas_1_peso = 0

img_negra = np.zeros_like(img_original)

#--- Repito parte del proceso, sumando la informacion de las divisiones de las areas y las diferencio con colores 
k = -1
for contorno in contornos:
        k += 1
        elipse = cv2.fitEllipse(contorno)
        excentricidad = elipse[1][0] / elipse[1][1]
        if excentricidad > 0.95:
            monedas.append(contorno)
            area = cv2.contourArea(contorno)
            if area <= 90000:
                monedas_10_centavos += 1
                cv2.drawContours(img_final, contornos, k, (0,0,255),3)
            elif 90000 < area <= 107000:
                monedas_50_centavos += 1
                cv2.drawContours(img_final, contornos, k, (0,255,0),3)
            else:
                monedas_1_peso += 1
                cv2.drawContours(img_final, contornos, k, (255,0,0),3)
        else:
            dados.append(contorno)
            cv2.drawContours(img_final, contornos, k, (200,200,100),3)
            cv2.drawContours(img_negra, contornos, k, (255, 255, 255), thickness=cv2.FILLED)
plt.figure(), plt.imshow(img_final), plt.show(block=False)

print(f"Hay {monedas_10_centavos} monedas de 10 centavos, {monedas_50_centavos} de 50 centavos y {monedas_1_peso} de 1 peso. Y {len(dados)} dados.")


# --- A la imagen con los contornos de los dados le agrego la original solo en esas partes.
plt.figure(), plt.imshow(img_negra), plt.show(block=False)
img_pintada = cv2.bitwise_and(img_original, img_negra)
plt.figure(), plt.imshow(img_pintada), plt.show(block=False)

# --- Paso a escala de grises
img_pintada_gris = cv2.cvtColor(img_pintada, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(img_pintada_gris, cmap='gray'), plt.show(block=False)

img_negra = np.zeros_like(img_original)


contornos_dados, _ = cv2.findContours(img_pintada_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def procesamiento(img):
    
    # Repito los pasos del principio para detectar los contornos dentro de cada dado
    # --- Suavizo
    img_fil2 = cv2.medianBlur(region_recortada_gris, 5)
    #plt.figure(), plt.imshow(img_fil, cmap='gray'), plt.show(block=False)

    # --- Filtro pasabajo
    img_pasabajo2 = cv2.blur(img_fil2, (5, 5))
    #plt.figure(), plt.imshow(img_fil, cmap='gray'), plt.show(block=False)

    # --- Detección de bordes con Canny
    bordes2 = cv2.Canny(img_pasabajo2, 10, 80)
    #plt.figure(), plt.imshow(bordes2, cmap='gray'), plt.show(block=False)

    return bordes2

# --- Repito el bucle de todos los contornos pero me enfoco en los dados
k = -1
m = 0 # Con este contador voy a seguir a cada dado
for contorno in contornos_dados:
    m += 1
    x, y, w, h = cv2.boundingRect(contorno)
    region_recortada = img_original[y:y + h, x:x + w]
    region_recortada_gris = cv2.cvtColor(region_recortada, cv2.COLOR_BGR2GRAY)

    bordes2 = procesamiento(region_recortada_gris)
    # --- Genero la máscara
    num_labels2, etiquetas2 = cv2.connectedComponents(bordes2)
    mascara2 = np.zeros_like(bordes2)

    for label in range(1, num_labels2):
        mascara2[etiquetas2 == label] = 255
    #plt.figure(), plt.imshow(mascara2, cmap='gray'), plt.show(block=False)

    # --- Cierro los circulos dilatando
    kernel2 = np.ones((7,7), np.uint8)
    dilatado2 = cv2.dilate(bordes2, kernel2, iterations=1)
    #plt.figure(), plt.imshow(dilatado2, cmap='gray'), plt.show(block=False)

    contornos2, _ = cv2.findContours(dilatado2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bordes_imagen2 = np.zeros_like(dilatado2)

    # Voy a "elegir" los contornos que yo necesito para analizar, filtrando por area
    area_minima = 2700
    area_maxima = 2950
    puntos = []

    for i, c in enumerate(contornos2):
        area_contorno = cv2.contourArea(c)
        if area_minima < area_contorno < area_maxima:
            cv2.drawContours(bordes_imagen2, [c], -1, 255, 1)
            puntos.append(c)
    numero_dado = len(puntos)
    etiqueta = f'Numero de dado: {numero_dado}'
    cv2.putText(bordes_imagen2, etiqueta, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    plt.figure(), plt.imshow(bordes_imagen2, cmap='gray'), plt.show(block=False)

