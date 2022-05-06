import cv2
from ultilities import get_contours, warp_image, calculate_distance, reorder_corner_points, de_shadow


w_a4 = 210
h_a4 = 297
scale = 3

cap = cv2.VideoCapture(0)
# codec = 0x47504A4D  # MJPG
cap.set(cv2.CAP_PROP_FPS, 30.0)
# cap.set(cv2.CAP_PROP_FOURCC, codec)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    contours = get_contours(frame, min_area=10000, draw_cont=False, show_canny=False)

    if len(contours) != 0:
        biggest_rect = contours[0][1]
        warp_img = warp_image(frame, biggest_rect, width=scale*w_a4, height=
                            scale*h_a4)
        contours2 = get_contours(warp_img, canny_threshold=(50, 100), min_area=100, draw_cont=False, show_canny=False)
        if len(contours2) != 0:
            for cont in contours2:
                points = reorder_corner_points(cont[1])
                # calculate width and height of object
                width_obj = calculate_distance(points[0][0], points[1][0]) // scale
                height_obj = calculate_distance(points[0][0], points[2][0]) // scale
                # display dimension of obj
                cv2.arrowedLine(warp_img, points[0][0], points[1][0], color=(0, 255, 0), thickness=3, tipLength=0.05)
                cv2.arrowedLine(warp_img, points[0][0], points[2][0], color=(0, 255, 0), thickness=3, tipLength=0.05)
                x, y, w, h = cont[2]
                cv2.putText(warp_img, f'{width_obj} mm', (x + w // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255), thickness=2)
                cv2.putText(warp_img, f'{height_obj} mm', (x - 70, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255), thickness=2)
        cv2.imshow('A4', warp_img)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()