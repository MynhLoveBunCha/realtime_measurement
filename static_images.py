import cv2
import os
from ultilities import (
    get_contours,
    warp_image,
    calculate_distance,
    reorder_corner_points,
)

# test_img_path = 'test_images/test3.jpg'
w_a4 = 210
h_a4 = 297
scale = 3
for path in os.listdir("test_images"):
    test_img = cv2.imread("test_images/" + path)

    # test_img = de_shadow(test_img)
    contours = get_contours(test_img, min_area=50000, draw_cont=False, show_canny=False)

    if len(contours) != 0:
        biggest_rect = contours[0][1]
        warp_img = warp_image(
            test_img, biggest_rect, width=scale * w_a4, height=scale * h_a4
        )
        contours2 = get_contours(
            warp_img,
            canny_threshold=(50, 100),
            min_area=100,
            draw_cont=False,
            show_canny=False,
        )
        if len(contours2) != 0:
            for cont in contours2:
                points = reorder_corner_points(cont[1])
                # calculate width and height of object
                width_obj = calculate_distance(points[0][0], points[1][0]) // scale
                height_obj = calculate_distance(points[0][0], points[2][0]) // scale
                # display dimension of obj
                cv2.arrowedLine(
                    warp_img,
                    points[0][0],
                    points[1][0],
                    color=(0, 255, 0),
                    thickness=3,
                    tipLength=0.05,
                )
                cv2.arrowedLine(
                    warp_img,
                    points[0][0],
                    points[2][0],
                    color=(0, 255, 0),
                    thickness=3,
                    tipLength=0.05,
                )
                x, y, w, h = cont[2]
                cv2.putText(
                    warp_img,
                    f"{width_obj} mm",
                    (x + w // 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.putText(
                    warp_img,
                    f"{height_obj} mm",
                    (x - 70, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )
        cv2.imshow("A4", warp_img)

    display_test = cv2.resize(
        test_img, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA
    )
    cv2.imshow("test", display_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
