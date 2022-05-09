######### Author : TAIZIERES Benjamin copyright 2022
import cv2
import numpy as np
import skimage
import imutils
from os import listdir
from os.path import isfile, join
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

# Déclaration variable contenant le dossier d'entré d'image
mypath='/home/debian/Documents/test/PROJET_GUADELOUP_MASTER/1987_PROCESS_CONVERT_JPG/'
# Test afin de recuperer uniquement la liste des images (seulement un fichier)
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
# Variable contenant la liste des images
images = np.empty(len(onlyfiles), dtype=object)

# Boucle de traitement afin de selectionner une image puis déffectuer le traitement avant de passer a une autre
for n in range(0, len(onlyfiles)):

    # Affichage en console
    print ("Traitement de :" +  join(mypath,onlyfiles[n]) )

    # Chargement d'une image, convertion à l'echelle de gris et application de la methode OTSU
    image = cv2.imread( join(mypath,onlyfiles[n]) )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Calculer la distance euclidienne de chaque pixel binaire
    # au pixel zéro le plus proche puis trouver les bords
    distance_map = ndimage.distance_transform_edt(thresh)
    # La variable min_distance permet de regler la finesse de détourage
    local_max = peak_local_max(distance_map, indices=False, min_distance=7, labels=thresh)


    # Analyse des composants de l'image avant traitement par l'algo Wathershed
    markers = ndimage.label(local_max, structure=np.ones((3, 3,)))[0]
    labels = watershed(-distance_map, markers, mask=thresh)

    # Itération de chaque labels
    total_area = 0
    for label in np.unique(labels):
        if label == 0:
            continue

        # Creation d'un masque
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Cherche les contours avant de les determiner puis les tracer en vert
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        total_area += area
        cv2.drawContours(image, [c], -1, (36,255,12), 2)

    # Affichage en console de la surface
    print(total_area)

    # Permet si décommenté de mettre d'afficher l'image 
    #cv2.imshow('image', image)

    # Récuperation du nom de l'image en cours de traitement
    nom = join(mypath,onlyfiles[n])

    # Sauvegarde de l'image dans le dossier de sortie
    cv2.imwrite('/home/debian/Documents/test/PROJET_GUADELOUP_MASTER/1969_PROCESS_CONVERT_JPG/results/' + nom.split("/")[7], image)
    
    # Permet si décommenté de mettre en pause le traitement
    #cv2.waitKey()

    # Affichage en console pour suivi
    print ("Fin de traitement")

# Affichage en console pour suivi
print ("Fin Script")