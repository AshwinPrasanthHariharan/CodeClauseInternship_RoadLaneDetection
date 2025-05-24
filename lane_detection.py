import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    triangle = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.5 * width), int(0.6 * height))
    ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined

cap = cv2.VideoCapture('road_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output = process_frame(frame)
    cv2.imshow('Lane Detection', output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
