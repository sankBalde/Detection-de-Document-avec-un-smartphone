# Detection de Document avec un smartphone

## Vue d'ensemble
Ce projet est conçu pour détecter et suivre des documents dans des séquences vidéo à l'aide d'OpenCV.
La solution inclut plusieurs fonctionnalités telles que la détection de documents, 
la transformation de perspective et le traitement vidéo pour suivre les documents en temps réel.
![doc_det.gif](videos%2Fdoc_det.gif)

## Dépendances
Pour exécuter le code, assurez-vous d'avoir les dépendances suivantes installées :
- OpenCV
- NumPy
- Matplotlib

Vous pouvez installer ces dépendances en utilisant pip :
```sh
pip install opencv-python numpy matplotlib
```

## Utilisation

Le projet contient plusieurs fonctions pour détecter et suivre des documents dans des images et des vidéos.

Fonctions principales

is_same_document(bbox, frame1, frame): Vérifie si un document dans une région d'intérêt (ROI) reste le même dans deux frames différents.

detect_document_rectangle(image): Détecte le rectangle englobant le plus grand dans une image.

detect_document_contour(image): Détecte le contour du plus grand quadrilatère dans une image.

perspective_transform_with_largest_contour(im_frame): Applique une transformation de perspective sur le plus grand quadrilatère détecté dans une image.

capture_and_detect(): Capture des images de la caméra en temps réel et détecte les documents.

capture_video_tracking(video_in_path, video_out_path): Traite une vidéo pour détecter et suivre des documents et enregistre la vidéo de sortie.

## Ressources:
REAL-TIME DOCUMENT DETECTION IN SMARTPHONE VIDEOS
De Elodie Puybareau, Thierry Géraud 

## Auteur
Abdoulaye Baldé

