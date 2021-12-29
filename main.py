import cv2 as cv
import numpy as np
import os
from time import time
from WindowCapture import WindowCapture
from vision import findClickPositions

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Find the game window
wincap = WindowCapture('BlueStacks 1')

while(True):
    fps_timer = time()

    # screenshot = wincap.get_screenshot()

    # Hard coded image for testing...
    screenshot = cv.imread('fv_screen3.jpg', cv.IMREAD_UNCHANGED)

    findClickPositions(screenshot, 'fv_needle6.jpg', threshold=0.5, method=cv.TM_CCOEFF_NORMED, debug_mode='both')
    # findClickPositions(screenshot, 'fv_needle6.jpg', threshold=0.5, method=cv.TM_CCOEFF_NORMED, debug_mode='both')
    # findClickPositions(screenshot, 'fv_needle6.jpg', threshold=0.01, method=cv.TM_SQDIFF_NORMED, debug_mode='both')

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    print('FPS {}' . format(1 / (time() - fps_timer)))

print('Done...')