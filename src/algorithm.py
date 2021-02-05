from board import Board
from extract import *
#from perspective import *
from util import showImage, drawPerspective, drawBoundaries, drawLines, drawPoint, drawContour, randomColor
from line import Line

import random
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

extract_width=512 #image width
extract_height=512 #image height

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

nvertical = 9 #chess board vertical lines
nhorizontal = 9 #chess board horizontal lines



#load gray image
def load_image(filename):
    image = cv2.imread(filename)
    return image

#get the edge contours from an RGB image
def get_contours(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #gray
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(im_bw,  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
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
        tmp = np.zeros(image.shape, np.uint8)
        drawContour(tmp, points, (255,), 1)

        # gaussian blur
        tmp = cv2.GaussianBlur(tmp,(kernel_size, kernel_size),0)

        # canny
        edge = cv2.Canny(tmp, 50, 150)

        edges.append(edge)

    return edges

# get perspective from edge contours
def get_perspective_from_contours(image):
    for t in range(threshold_max, threshold_min, -threshold_step):
        # Run Hough on edge detected image
        lines = cv2.HoughLines(image, rho, theta, t, np.array([]),
                            min_line_length, max_line_gap)
        
        perspective = get_perspective_from_lines(lines)
        if perspective is not None:
            return perspective
    
    raise("error: cannot extract perspective")

# get 8x8 grid from edge contours
def get_grid_from_contours(image, close_threshold_v, close_threshold_h):
    for t in range(threshold_max, threshold_min, -threshold_step):
        # Run Hough on edge detected image
        # print("get_grid_from_contours t ", t)
        lines = cv2.HoughLines(image, rho, theta, t, np.array([]),
                            min_line_length, max_line_gap)
        
        #if len(lines) > 100:
        #    break
        
        grid = get_grid_from_lines(lines, close_threshold_v, close_threshold_h, image)
        if grid is not None:
            return grid
    
    raise("error: cannot extract grid")

# get perspective(square board) from lines
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

# get grid from lines
def get_grid_from_lines(lines, close_threshold_v, close_threshold_h, image):
    q_lines = [Line(l[0][0], l[0][1]) for l in lines]
    #print("get_grid_from_lines 1", lines.shape)
    horizontal, vertical = partitionLines(q_lines)
    vertical = filterCloseLines(vertical, horizontal=False, threshold=close_threshold_v)
    horizontal = filterCloseLines(horizontal, horizontal=True, threshold=close_threshold_h)

    vertical = [v for v in filter(lambda x:x.isStraight(), vertical)]
    horizontal = [h for h in filter(lambda x:x.isStraight(), horizontal)]
    

    #print("get_grid_from_lines 2",len(vertical), len(horizontal))
    if len(vertical) >= nvertical and len(horizontal) >= nhorizontal:
        grid = (vertical, horizontal)

        drawLines(image, vertical, color=(255,255,0), thickness=2)
        drawLines(image, horizontal, color=(255,255,0), thickness=2)
        
        #---------draw grid
        print("grid are drawn")
        plt.imshow(image)
        plt.show()
        print("grid are drawn end")

        return grid


def get_boards_from_perspective(image, perspective):
    b = extractPerspective(image, perspective, extract_width, extract_height)
    
    plt.imshow(b)
    plt.show()
    
    w, h, _ = image.shape
    close_threshold_v = (w / nvertical) / 4
    close_threshold_h = (h / nhorizontal) / 4

    im_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    
    thresh, im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_canny = cv2.Canny(im_bw, 50, 150, apertureSize=3)

    plt.imshow(im_canny,cmap="gray")
    plt.show()

    #get every grid
    grid = get_grid_from_contours(im_canny, close_threshold_v, close_threshold_h)

    #get tiles of a board
    tiles = extractTiles(b, grid, 64, 64)

    return Board(tiles, 8, 8)



if __name__ == "__main__":
    img = load_image("../examples/1.jpg")
    edges = get_contours(img)
    perspective = get_perspective_from_contours(edges[0])
    print(perspective)

    board =get_boards_from_perspective(img, perspective)
    
