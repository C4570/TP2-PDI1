import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- Cargo Imagen ------------------------------------------
plt.close('all')
I = cv2.imread(f"Patentes/img07.png") 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I, title="Imagen Original")


# --- Paso a escalas de grises ------------------------------
Ig = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
imshow(Ig, title="Imagen en escala de grises")

# --- Binarizo ---------------------------------------------
th, Ibw = cv2.threshold(Ig, 121, 255, cv2.THRESH_BINARY)    
imshow(Ibw, title="Umbralado")


# --- Elementos conectados ---------------------------------
connectivity = 6
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw, connectivity, cv2.CV_32S)  
imshow(labels, title="Componentes conectados")


# *** Observo las áreas de todos los objetos **********
areas = [st[cv2.CC_STAT_AREA] for st in stats]
areas_sorted = sorted(areas)
print(areas_sorted)
for ii, vv in enumerate(areas_sorted):
    print(f"{ii:3d}): {vv:8d}")
# *****************************************************

# --- Filtro por area ---------------------------------------------------------------
Ibw_filtArea = Ibw.copy()
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_AREA]>100) or (stats[jj, cv2.CC_STAT_AREA]<15):
        Ibw_filtArea[labels==jj] = 0
imshow(Ibw_filtArea, title="Filtrado por Area (imagen binaria)")

# --- Filtro por relacion de aspecto ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw_filtArea, connectivity, cv2.CV_32S)  
imshow(labels, title="Componentes Conectados")

Ibw_filtAspect = Ibw_filtArea.copy()
for jj in range(1,num_labels):
    rel_asp = stats[jj, cv2.CC_STAT_HEIGHT] / stats[jj, cv2.CC_STAT_WIDTH]
    print(f"{jj:3d}) {rel_asp:5.2f}")
    if (rel_asp<1.5) or (rel_asp >3.0):
        Ibw_filtAspect[labels==jj] = 0
imshow(Ibw_filtAspect, title="Filtrado por relación de aspecto")


# --- Resultado parcial ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw_filtAspect, connectivity, cv2.CV_32S)  
imshow(labels, title="Componentes Conectados")
Ipatente = cv2.merge((Ibw_filtAspect, Ibw_filtAspect, Ibw_filtAspect))

# *** Analizo las relaciones de aspecto de todos los objetos *****************************
for ii in range(1,num_labels):
    rel_asp = stats[ii, cv2.CC_STAT_HEIGHT] / stats[ii, cv2.CC_STAT_WIDTH]
    print(f"{ii:3d}) {rel_asp:5.2f}")
 # ***************************************************************************************


# --- Corroboro cercania de otro caracter ----------------------------
DIST_TH = 20
Ipatente_cercania = Ibw_filtAspect.copy()
for ii in range(1, num_labels):
    ch = centroids[ii,:]       # Centroide del caracter
    # --- Obtengo los centroides de los demás caracteres ---------------------------
    objs = np.delete(centroids.copy(), ii, axis=0)  # Elimino centroide actual
    objs = np.delete(objs, 0, axis=0)               # Elimino centroide del fondo
    # --- Calculo distancias -------------------------------------------------------
    aux = objs - ch
    dists = np.sqrt(aux[:,0]**2 + aux[:,1]**2)
    if not any(dists < DIST_TH):
        # print(f"{ii} --> Eliminado")
        Ipatente_cercania[labels==ii] = 0
imshow(Ipatente_cercania, title="Filtrado por cercanía")

# --- Resultado final ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ipatente_cercania, connectivity, cv2.CV_32S)  
Ifinal = cv2.merge((Ipatente_cercania, Ipatente_cercania, Ipatente_cercania))
for ii in range(1,num_labels):
    cv2.rectangle(Ifinal, tuple(stats[ii,0:2]), tuple(stats[ii,0:2]+stats[ii,2:4]), (255,0,0), 1)
imshow(Ifinal, title="Resultado Final: Caraceteres")    





#img2 lista con 121
#img4 lista con 121
#img5 lista con 121
#img6 lista con 121
#img7 lista con 121
#img9 lista con 121
#img10 lista con 121 

#img1 con 145 si se lo abajo a la binarizacion me lo toma como parte del borde de la patente
#img11 con 145 si se lo abajo a la binarizacion me lo toma como parte del borde de la patente
#img08 con 145 si se lo abajo a la binarizacion me lo toma como parte del borde de la patente

#img12 con 113 si le subo la binarizacion se me divide un caracter a la mitad
# img3 me falta la primera letra pero si se lo subo me corta las letras y si se lo bajo me lo toma como parte del borde de la patente
