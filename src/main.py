import cv2
import numpy as np
import itertools
import time
from utils import *


def is_same_document(bbox, frame1, frame):
    x, y, w, h = bbox

    # Extraire la ROI et convertir en HSV
    roi = frame1[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Créer un masque pour filtrer les pixels non pertinents
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # Calculer l'histogramme de la ROI
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Critères de terminaison pour l'algorithme de MeanShift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Backprojection de l'histogramme sur l'image courante
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Appliquer l'algorithme de MeanShift pour obtenir la nouvelle position
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

    # Dessiner le rectangle de suivi sur l'image
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
    return img2, [x, y, w, h]
def detect_document_rectangle(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Utiliser la méthode de détection de contours par Canny
    edged = cv2.Canny(blurred, 75, 200)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser une variable pour le plus grand rectangle
    max_area = 0
    best_rect = None

    # Boucler sur les contours pour trouver le plus grand rectangle
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Si l'aire du rectangle est plus grande que la précédente, mettre à jour
        if area > max_area:
            max_area = area
            best_rect = (x, y, x, y)

    # Si un rectangle a été trouvé
    if best_rect is not None:
        x, y, w, h = best_rect
        # Dessiner le rectangle du document sur l'image originale
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        return image, [x, y, w, h]
    else:
        print("Aucun document n'a été détecté.")

    return image, []


def detect_document_contour(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Utiliser la méthode de détection de contours par Canny
    edged = cv2.Canny(blurred, 75, 200)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser une variable pour le plus grand quadrilatère
    max_area = 0
    best_contour = None

    # Boucler sur les contours pour trouver le plus grand rectangle
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Si l'aire du rectangle est plus grande que la précédente, mettre à jour
        if area > max_area:
            max_area = area
            best_contour = contour

    # Si un quadrilatère a été trouvé
    if best_contour is not None:
        # Dessiner le contour du document sur l'image originale
        cv2.drawContours(image, [best_contour], -1, (255, 0, 0), 10)
    else:
        print("Aucun document n'a été détecté.")

    return image


def perspective_transform_with_largest_contour(im_frame):
    if im_frame is None:
        print("Erreur lors du chargement de l'image")
        return

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détecter les contours
    edges = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser une variable pour le plus grand quadrilatère
    max_area = 0
    best_rect = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Si l'aire du rectangle est plus grande que la précédente, mettre à jour
        if area > max_area:
            max_area = area
            best_rect = np.array([x, y, w, h])

    # Si un contour valide est trouvé
    if best_rect is not None:
        x, y, w, h = best_rect
        pts_src = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')

        # Organiser les points du quadrilatère
        s = pts_src.sum(axis=1)
        diff = np.diff(pts_src, axis=1)

        top_left = pts_src[np.argmin(s)]
        bottom_right = pts_src[np.argmax(s)]
        top_right = pts_src[np.argmin(diff)]
        bottom_left = pts_src[np.argmax(diff)]

        pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # Définir les points de destination pour la transformation de perspective
        width, height = im_frame.shape[1], im_frame.shape[0]
        pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # Calculer la transformation de perspective
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Appliquer la transformation de perspective
        warped = cv2.warpPerspective(im_frame, M, (width, height))

        return warped

    return im_frame
def capture_and_detect():
    cap = cv2.VideoCapture(0)  # Ouvre la caméra (généralement 0 ou 1 pour les webcams)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : Impossible de lire une image de la caméra.")
            break

        # Détecter le document dans le frame capturé
        detected_frame = perspective_transform_with_largest_contour(frame)

        # Afficher le flux vidéo avec le document détecté
        cv2.imshow('Camera - Document Detection', detected_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quitter si la touche 'q' est pressée
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_video_tracking(video_in_path, video_out_path):
    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Lire la première frame pour obtenir les dimensions
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la vidéo.")
        return

    # Définir le codec et créer un objet VideoWriter pour enregistrer la vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    # Initialiser les variables pour le calcul des FPS
    prev_time = time.time()
    fps = 0
    bbox1 = None
    while ret:
        # Calculer le temps écoulé entre les frames
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time

        # Calculer le FPS
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Détecter le document dans le frame capturé
        markers, image_blacked, image_frame = preprocessing(frame)
        cop_im, top_left, top_right, bottom_left, bottom_right = find_bbox_in_document(markers, image_blacked, image_frame)
        #detected_frame, bbox = detect_document_rectangle(frame)

        # Ajouter le FPS au frame
        cv2.putText(cop_im, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Écrire le cadre traité dans le fichier vidéo
        out.write(cop_im)

        # Afficher le frame traité
        #cv2.imshow('Video - Document Detection', detected_frame)

        # Lire le prochain frame
        ret, frame = cap.read()

        # Quitter la boucle si 'ESC' est pressée (optionnel, peut être désactivé)
        if cv2.waitKey(1) & 0xFF == 27:  # Touche ESC
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


#capture_video_tracking("../videos/find_me.mp4", "../videos/best2_detection_video_two.mp4")