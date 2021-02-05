from board import Board
from extract import *
#from perspective import *
#from util import showImage, drawPerspective, drawBoundaries, drawLines, drawPoint, drawContour, randomColor
from line import Line

import random
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

extract_width=400 #image width
extract_height=400 #image height

kernel_size = 5 #kernal for Gaussian blur

canny_threshhold_low = 50 # canny 
canny_threshhold_high = 150 #canny

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid

threshold_max = 150  # minimum number of votes (intersections in Hough grid cell)
threshold_min = 10
threshold_step = 10

min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments



#load gray image
def load_image(filename):
    image = cv2.imread(filename)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

def get_contours(image):
    contours, hierarchy = cv2.findContours(image,  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contour_ids = ignoreContours(image, contours, hierarchy)

    print("contour_ids",contour_ids)

    #create an empty image for contours
    img_contours = np.zeros(image.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, [contours[i] for i in contour_ids], -1, (0,255,0), 3)

    edges = []
    # loop all possible contours
    for c_id in contour_ids:
        points = contours[c_id]
        points = np.squeeze(points,1)

        # draw contours after filter
        yy, xx = image.shape
        tmp = np.zeros((yy, xx, 3), np.uint8)
        drawContour(tmp, points, (255,), 1)

        # gaussian blur
        tmp = cv2.GaussianBlur(tmp,(kernel_size, kernel_size),0)

        # canny
        edge = cv2.Canny(tmp, 50, 150)

        edges.append(edge)

    return edges

def get_perspective_from_contours(image):
    for t in range(threshold_max, threshold_min, -threshold_step):
        # Run Hough on edge detected image
        lines = cv2.HoughLines(image, rho, theta, t, np.array([]),
                            min_line_length, max_line_gap)
        
        perspective = get_perspective_from_lines(lines)
        if perspective is not None:
            return perspective
    
    raise("error: cannot extract perspective")

def get_perspective_from_lines(lines):
    p_lines = [Line(l[0][0], l[0][1]) for l in lines]
    (horizontal, vertical) = partitionLines(p_lines)
    vertical = filterCloseLines(vertical, horizontal=False)
    horizontal = filterCloseLines(horizontal, horizontal=True)
    
    if len(vertical) == 2 and len(horizontal) == 2:
        grid = (vertical, horizontal)

        if vertical[0].getCenter()[0] > vertical[1].getCenter()[0]:
            v2, v1 = vertical
        else:
            v1, v2 = vertical

        if horizontal[0].getCenter()[1] > horizontal[1].getCenter()[1]:
            h2, h1 = horizontal
        else:
            h1, h2 = horizontal

        perspective = (h1.intersect(v1),
                    h1.intersect(v2),
                    h2.intersect(v2),
                    h2.intersect(v1))

        return perspective

    else:
        return None

if __name__ == "__main__":
    edges = get_contours(load_image("../examples/1.jpg"))
    perspective = get_perspective_from_contours(edges[0])
    print(perspective)

    
