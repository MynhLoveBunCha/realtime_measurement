from blend_modes import divide
import cv2
import numpy as np


def de_shadow(image):
    # splitting the image into channels
    bA = image[:, :, 0]
    gA = image[:, :, 1]
    rA = image[:, :, 2]

    # dialting the image channels individually to spead the text to the background
    dilated_image_bB = cv2.dilate(bA, np.ones((7,7), np.uint8))
    dilated_image_gB = cv2.dilate(gA, np.ones((7,7), np.uint8))
    dilated_image_rB = cv2.dilate(rA, np.ones((7,7), np.uint8))

    # blurring the image to get the backround image
    bB = cv2.medianBlur(dilated_image_bB, 21)
    gB = cv2.medianBlur(dilated_image_gB, 21)
    rB = cv2.medianBlur(dilated_image_rB, 21)

    # blend_modes library works with 4 channels, the last channel being the alpha channel
    # so we add one alpha channel to our image and the background image each
    image = np.dstack((image, np.ones((image.shape[0], image.shape[1], 1))*255))
    image = image.astype(float)
    dilate = [bB,gB,rB]
    dilate = cv2.merge(dilate)
    dilate = np.dstack((dilate, np.ones((image.shape[0], image.shape[1], 1))*255))
    dilate = dilate.astype(float)

    # now we divide the image with the background image
    # without rescaling i.e scaling factor = 1.0
    blend = divide(image,dilate,1.0)
    blendb = blend[:, :, 0]
    blendg = blend[:, :, 1]
    blendr = blend[:, :, 2]
    blend_planes = [blendb,blendg,blendr]
    blend = cv2.merge(blend_planes)
    # blend = blend*0.85
    blend = np.uint8(blend)

    # returning the shadow-free image
    return blend


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest, max_area


def get_contours(bgr_image, canny_threshold=(100, 200), min_area=500, shadow_free=False, show_canny=False, draw_cont=False):
    # preprocess img
    if shadow_free:
        no_shadow_img = de_shadow(bgr_image)
        gray_img = cv2.cvtColor(no_shadow_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    # find edges and more preprocess
    canny_img = cv2.Canny(blur_img, canny_threshold[0], canny_threshold[1])
    kernel = np.ones((5, 5))
    dilate_img = cv2.dilate(canny_img, kernel, iterations=3)
    erode_img = cv2.erode(dilate_img, kernel, iterations=2)
    if show_canny:
        cv2.imshow('Cany edge dectection', cv2.resize(erode_img, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA))
    # find contours
    contours, _ = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > min_area:
            peri = cv2.arcLength(cont, closed=True)
            approx_corners = cv2.approxPolyDP(cont, 0.02*peri, closed=True)
            bbox = cv2.boundingRect(approx_corners)
            if len(approx_corners) == 4:
                final_contours.append((area, approx_corners, bbox, cont))
    final_contours = sorted(final_contours, key=lambda x:x[0], reverse=True)
    if draw_cont:
        for cont in final_contours:
            cv2.drawContours(bgr_image, cont[3], -1, (0 ,0, 255), 7)
    return final_contours


def reorder_corner_points(points):
    new_points = np.zeros_like(points)
    reshape_points = points.reshape((4, 2))  # reshape into 2-D array
    # Find the upper left corner and the lower right corner
    add = np.sum(reshape_points, axis=1)
    new_points[0] = reshape_points[np.argmin(add)]
    new_points[3] = reshape_points[np.argmax(add)]
    # Find the upper right corner and the lower left corner
    diff = np.diff(reshape_points, axis=1)
    new_points[1] = reshape_points[np.argmin(diff)]
    new_points[2] = reshape_points[np.argmax(diff)]
    return new_points


def warp_image(img, corner_points, width, height, pad=10):
    points = reorder_corner_points(corner_points)
    # Find perspective transform matrix
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # warping perspective
    warp_img = cv2.warpPerspective(img, matrix, (width, height))
    warp_img = warp_img[pad:warp_img.shape[0] - pad, pad:warp_img.shape[1] - pad]
    return warp_img


def calculate_distance(pts1, pst2):
    return ((pst2[0] - pts1[0])**2 + (pst2[1] - pts1[1])**2)**0.5