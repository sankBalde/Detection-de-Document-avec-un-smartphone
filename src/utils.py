import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_segments(image, segments, color=(0, 0, 255), thickness=20):
    for segment in segments:
        if len(segment) >= 2:
            cv2.line(image, segment[0], segment[1], color, thickness)
    return image

def classify_segments_in_two(segments):
    horizontal_segments = []
    vertical_segments = []

    for segment in segments:
        (x1, y1), (x2, y2) = segment
        if abs(y2 - y1) < abs(x2 - x1):  # Horizontal segment
            horizontal_segments.append(segment)
        else:  # Vertical segment
            vertical_segments.append(segment)

    return horizontal_segments, vertical_segments

def bresenham(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points




def calculate_segment_attributes(segment, watershed_image):
    (x1, y1), (x2, y2) = segment
    length = np.linalg.norm(np.array((x1, y1)) - np.array((x2, y2)))
    #bre = bresenham(x1, y1, x2, y2)
    #distance_to_watershed = np.mean([watershed_image[y, x] for x, y in bre])
    distance_to_watershed = 1
    return length, distance_to_watershed
def compute_gradient(channel, size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    dilated = cv2.dilate(channel, kernel)
    eroded = cv2.erode(channel, kernel)
    return dilated - eroded
def get_intersection_points(x1, y1, x2, y2, width, height):
    points = []
    if x2 != x1:
        y = int(y1 + (0 - x1) * (y2 - y1) / (x2 - x1))
        if 0 <= y < height:
            points.append((0, y))
        y = int(y1 + (width - 1 - x1) * (y2 - y1) / (x2 - x1))
        if 0 <= y < height:
            points.append((width - 1, y))
    if y2 != y1:
        x = int(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
        if 0 <= x < width:
            points.append((x, 0))
        x = int(x1 + (height - 1 - y1) * (x2 - x1) / (y2 - y1))
        if 0 <= x < width:
            points.append((x, height - 1))
    return points

def filter_segment_Image_shape(segments, width, height):
  filtered_segments = []
  for segment, length, energy in segments:
    p1, p2 = segment

    points= get_intersection_points(p1[0], p1[1], p2[0], p2[1], width, height)
    if len(points) == 2:
      filtered_segments.append(segment)
  return filtered_segments

def classify_segments_in_Four(segments_hori, segments_vert, width, height):
    top_segments = []
    bottom_segments = []
    left_segments = []
    right_segments = []
    margin = 50
    for segment in segments_hori:

        (x1, y1), (x2, y2) = segment
        # Éliminer les segments proches des bords de l'image
        if (x1 < margin and x2 < margin) or (x1 > width - margin and x2 > width - margin):
            continue
        if (y1 < margin and y2 < margin) or (y1 > height - margin and y2 > height - margin):
            continue
        # Segment horizontal
        if y1 < height // 2:
            top_segments.append(segment)
        else:
            bottom_segments.append(segment)

    for segment in segments_vert:
        (x1, y1), (x2, y2) = segment
        # Éliminer les segments proches des bords de l'image
        if (x1 < margin and x2 < margin) or (x1 > width - margin and x2 > width - margin):
            continue
        if (y1 < margin and y2 < margin) or (y1 > height - margin and y2 > height - margin):
            continue
        # Segment vertical
        if x1 < width // 2:
            left_segments.append(segment)
        else:
          right_segments.append(segment)


    return top_segments, bottom_segments, left_segments, right_segments


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def preprocessing(image_frame):
    lab = cv2.cvtColor(image_frame, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    L_closed = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel_close)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a_eroded = cv2.erode(a, kernel_erode)
    lab_processed = cv2.merge([L_closed, a_eroded, b])

    L, a, b = cv2.split(lab_processed)
    # delta = dilation - erosion
    delta_L = compute_gradient(L)
    delta_a = compute_gradient(a)
    delta_b = compute_gradient(b)
    sum_delta = delta_L + delta_a + delta_b
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient_closed = cv2.morphologyEx(sum_delta, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(gradient_closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_frame, markers)
    image_blacked = np.zeros_like(image_frame)
    image_blacked[markers == -1] = [255, 0, 0]

    return markers, image_blacked, image_frame

def find_bbox_in_document(markers, image_blacked, image_frame):
    image_blacked_gray = cv2.cvtColor(image_blacked, cv2.COLOR_BGR2GRAY)
    # Appliquer la transformée de Hough pour détecter les lignes
    lines = cv2.HoughLines(image_blacked_gray, 1, np.pi / 180, 200)
    segments = []

    # Obtenir les dimensions de l'image
    height, width = image_blacked_gray.shape[:2]

    # Découper les lignes détectées en segments (chunks)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        points = get_intersection_points(x1, y1, x2, y2, width, height)

        if len(points) == 2:
            segments.append(points)

    # Classifier les segments en horizontaux et verticaux
    horizontal_segments, vertical_segments = classify_segments_in_two(segments)

    # Calculer les attributs des segments
    vertical_segment_attributes = []
    horizontal_segment_attributes = []
    for segment in vertical_segments:
        length, distance_to_watershed = calculate_segment_attributes(segment, image_blacked_gray)
        vertical_segment_attributes.append((segment, length, distance_to_watershed))
    for segment in horizontal_segments:
        length, distance_to_watershed = calculate_segment_attributes(segment, image_blacked_gray)
        horizontal_segment_attributes.append((segment, length, distance_to_watershed))

    # Sélectionner les segments basés sur l'énergie U (simplification)
    selected_vertical_segments = sorted(vertical_segment_attributes, key=lambda x: x[1])[:10]

    selected_horizontal_segments = sorted(horizontal_segment_attributes, key=lambda x: x[1])[:10]

    selected_vertical_segments = filter_segment_Image_shape(selected_vertical_segments, width, height)
    selected_horizontal_segments = filter_segment_Image_shape(selected_horizontal_segments, width, height)

    top_segments, bottom_segments, left_segments, right_segments = classify_segments_in_Four(selected_horizontal_segments,
                                                                                     selected_vertical_segments, width,
                                                                                     height)
    # Tracer les segments sélectionnés et classifiés
    cop_im = image_frame.copy()
    cop_im = draw_segments(cop_im, top_segments)
    cop_im = draw_segments(cop_im, bottom_segments, color=(0, 255, 0))
    cop_im = draw_segments(cop_im, left_segments, color=(255, 255, 0))
    cop_im = draw_segments(cop_im, right_segments, color=(255, 0, 0))
    top_segment = top_segments[0] if top_segments else None
    bottom_segment = bottom_segments[0] if bottom_segments else None
    left_segment = left_segments[0] if left_segments else None
    right_segment = right_segments[0] if right_segments else None

    # Trouver les intersections des segments pour obtenir les coins du rectangle
    top_left = line_intersection(top_segment, left_segment) if top_segment and left_segment else None
    top_right = line_intersection(top_segment, right_segment) if top_segment and right_segment else None
    bottom_left = line_intersection(bottom_segment, left_segment) if bottom_segment and left_segment else None
    bottom_right = line_intersection(bottom_segment, right_segment) if bottom_segment and right_segment else None

    # Afficher les points des coins du rectangle
    print("Top left:", top_left)
    print("Top right:", top_right)
    print("Bottom left:", bottom_left)
    print("Bottom right:", bottom_right)
    if top_left:
        cv2.circle(cop_im, tuple(map(int, top_left)), 50, (0, 255, 0), -1)
    if top_right:
        cv2.circle(cop_im, tuple(map(int, top_right)), 50, (0, 255, 0), -1)
    if bottom_left:
        cv2.circle(cop_im, tuple(map(int, bottom_left)), 50, (0, 255, 0), -1)
    if bottom_right:
        cv2.circle(cop_im, tuple(map(int, bottom_right)), 50, (0, 255, 0), -1)
    return cop_im, top_left, top_right, bottom_left, bottom_right


