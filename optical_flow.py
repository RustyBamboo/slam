import numpy as np
from tinygrad import Tensor

from cv_tinygrad.cv_tinygrad import *

import cv2

import matplotlib.pyplot as plt

import sys

import imageio


def get_magnitude(u, v, scale=3):
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            counter += 1
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2) ** 0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg


def draw_quiver(u, v, beforeImg, scale=3):

    magnitudeAvg = get_magnitude(u, v, scale)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2) ** 0.5
            # draw only significant changes
            if magnitude > magnitudeAvg:
                angle = np.arctan2(dy, dx)

                r = 255 * np.cos(angle)
                g = 255 * np.cos(angle + 2 * np.pi / 3)
                b = 255 * np.cos(angle + 4 * np.pi / 3)

                # print(r,g,b, angle)

                cv2.arrowedLine(
                    beforeImg,
                    (j, i),
                    (int(j + dx), int(i + dy)),
                    color=(int(b), int(g), int(r)),
                    thickness=1,
                    tipLength=0.1,
                )


if __name__ == "__main__":
    save_video = False
    image_size = (500, 500)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
    cap = cv2.VideoCapture(fn)

    frames = []

    ret, previous_frame = cap.read()
    previous_frame = cv2.resize(previous_frame, image_size)
    previous_frame = Tensor(previous_frame)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, image_size)
        frame = Tensor(frame)

        u, v = horn_schunck(
            to_grayscale(previous_frame),
            to_grayscale(frame),
            alpha=2,
            num_iter=20,
            delta=0.1,
        )

        u = u.numpy()
        v = v.numpy()
        previous_frame = frame

        magnitude, angle = cv2.cartToPolar(u, v)

        mask = np.zeros_like(frame.numpy())
        mask[..., 1] = 255
        mask[..., 0] = angle * 180 / np.pi / 2

        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        img = frame.numpy()
        draw_quiver(u, v, img, scale=50)

        cv2.imshow("dense optical flow", rgb)
        cv2.imshow("vector", img)

        if save_video:
            vis = np.concat((rgb, img), axis=1)
            frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    if save_video:
        print("Saving gif...")
        imageio.mimsave("output/output.gif", frames, fps=30)
