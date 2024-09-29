import numpy as np
import cv2 as cv

# Remplacez 'chemin/vers/votre/video.mp4' par le chemin réel de votre fichier vidéo
cap = cv.VideoCapture('../videos/doc_track.mp4')

# Lire la première image de la vidéo
ret, frame = cap.read()

# Sélectionner la région d'intérêt (ROI) pour le document
bbox = cv.selectROI('select', frame, False)
print(bbox)
x, y, w, h = bbox

# Extraire la ROI et convertir en HSV
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# Créer un masque pour filtrer les pixels non pertinents
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# Calculer l'histogramme de la ROI
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Critères de terminaison pour l'algorithme de MeanShift
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# Définir le codec et créer un objet VideoWriter pour enregistrer la vidéo
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('../videos/output.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()

    if ret:
        # Convertir l'image courante en HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Backprojection de l'histogramme sur l'image courante
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Appliquer l'algorithme de MeanShift pour obtenir la nouvelle position
        ret, track_window = cv.meanShift(dst, (x, y, w, h), term_crit)

        # Dessiner le rectangle de suivi sur l'image
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        # Écrire le cadre traité dans le fichier vidéo
        out.write(img2)

        # Quitter la boucle si 'ESC' est pressé (optionnel, peut être désactivé)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

# Libérer les ressources
cap.release()
out.release()
cv.destroyAllWindows()