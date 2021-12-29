import cv2 as cv
import numpy as np

def findClickPositions(screenshot_image, needle_image_path, threshold = 0.5, method = cv.TM_CCOEFF_NORMED, debug_mode = None):
    points = []

    needle_image = cv.imread(needle_image_path, cv.IMREAD_UNCHANGED)

    needle_width = needle_image.shape[1]
    needle_height = needle_image.shape[0]

    result = cv.matchTemplate(screenshot_image, needle_image, method)

    if method == cv.TM_SQDIFF_NORMED:
        locations = np.where(result <= threshold)
    else:
        locations = np.where(result >= threshold)

    locations = list(zip(*locations[::-1])) # [x,y], [x,y]...

    rectangles = []
    for location in locations:
        rectangle = [int(location[0]), int(location[1]), needle_width, needle_height] # [x, y, h, w]
        rectangles.append(rectangle)
        rectangles.append(rectangle)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

    if len(rectangles):
        print('Found Needle...')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (255, 0, 0)
        marker_type = cv.MARKER_CROSS

        for (position_x, position_y, position_width, position_hight) in rectangles:
            center_x = position_x + int(position_width/2)
            center_y = position_y + int(needle_height/2)
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                top_left = (position_x, position_y)
                bottom_right = (position_x + position_width, position_y +  position_hight)
                cv.rectangle(screenshot_image, top_left, bottom_right, line_color, line_type)
            elif debug_mode == 'points':
                cv.drawMarker(screenshot_image, (center_x, center_y), marker_color, marker_type)
            elif debug_mode == 'both':
                top_left = (position_x, position_y)
                bottom_right = (position_x + position_width, position_y +  position_hight)
                cv.rectangle(screenshot_image, top_left, bottom_right, line_color, line_type)
                cv.drawMarker(screenshot_image, (center_x, center_y), marker_color, marker_type)

    if debug_mode:
        cv.imshow('Results', screenshot_image)

    return points