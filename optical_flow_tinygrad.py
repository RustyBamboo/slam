import numpy as np
from tinygrad import Tensor, dtypes
import os, sys

import cv2
from PIL import Image

from cv_tinygrad.cv_tinygrad import *

import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"

root_data_path = sys.argv[1]

root_dir = root_data_path+"/sequences/00"
pose_path = root_data_path+"/poses/00.txt"




class KITTIDataset:
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, "image_2")
        self.left_paths = sorted(
            [os.path.join(left_dir, imname) for imname in os.listdir(left_dir)]
        )

        if mode == "train":
            right_dir = os.path.join(root_dir, "image_3")
            self.right_paths = sorted(
                [os.path.join(right_dir, imname) for imname in os.listdir(right_dir)]
            )
            assert len(self.right_paths) == len(self.left_paths)
            self.transform = transform
            self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_paths[idx])
        if self.mode == "train":
            right_img = Image.open(self.right_paths[idx])
            sample = {"left_img": left_img, "right_img": right_img}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_img = self.transform(left_img)
            return left_img


dataset = KITTIDataset(root_dir, mode="train")


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

# print(cart2pol(u, v))

focal_length=718.8560
pp=(607.1928, 185.2157)

R = np.eye(3, dtype=np.float32)
t = np.zeros(shape=(3, 1), dtype=np.float32)

traj = np.zeros(shape=(600, 800, 3))


for i in range(1000):
    img1 = Tensor(np.array(dataset[i]["left_img"]))
    img2 = Tensor(np.array(dataset[i+1]["left_img"]))
    u, v = horn_schunck(to_grayscale(img1), to_grayscale(img2))

    u = u.numpy()
    v = v.numpy()


    xes = np.tile(np.arange(img1.shape[0]),(img1.shape[1],1))
    yes = np.tile(np.arange(img1.shape[1])[:,None],(1,img1.shape[0]))

    nxes = xes + u.T
    nyes = yes + v.T

    xy  = np.stack((yes,xes),axis = 2)
    nxy = np.stack((nyes,nxes),axis = 2)

    old_pts = np.reshape(xy,(-1, xy.shape[-1]))
    new_pts = np.reshape(nxy,(-1, nxy.shape[-1]))

    uv_pts = np.stack((u.T,v.T),axis = 2)
    uv_pts = np.reshape(uv_pts,(-1, uv_pts.shape[-1]))
    mag = np.sqrt(uv_pts[:,0]**2 + uv_pts[:,1]**2) > 0.5

    old_pts = old_pts[mag]
    new_pts = new_pts[mag]

    E, _ = cv2.findEssentialMat(
        new_pts,
        old_pts,
        focal_length,
        pp,
        cv2.RANSAC,
        0.999,
        1.0,
        None,
    )

    # R_old = R.copy()
    # t_old = t.copy()
    _, R_new, t_new, _ = cv2.recoverPose(
        E,
        old_pts,
        new_pts,
        R.copy(),
        t.copy(),
        focal=focal_length,
        pp=pp,
        # None,
    )

    # if (
        # abs(t[2][0]) > abs(t[0][0])
        # and abs(t[2][0]) > abs(t[1][0])
    # ):
    t = t + 1 * R.dot(t_new)
    R = R_new.dot(R)



    diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    adj_coord = np.matmul(diag, t)

    mono_coord = adj_coord.flatten()

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]

    # traj = cv.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
    traj = cv2.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

    cv2.imshow("trajectory", traj)

    magnitude, angle = cv2.cartToPolar(u, v)

    mask = np.zeros_like(img1.numpy())
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    # mask[..., 0] = 255

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("dense optical flow", rgb)

    img = img1.numpy()
    for point in old_pts:
        cv2.circle(img, center=tuple(point.astype(int)), radius=1, color=(255, 0, 0))

    for point in new_pts:
        cv2.circle(img, center=tuple(point.astype(int)), radius=1, color=(0, 0, 255))

    cv2.imshow("img", img)
    # print(magnitude, angle)
    # draw_quiver(u,v,img1.numpy())
    k = cv2.waitKey(1)

    if k == ord("q"):
        break
# closing all open windows
# cv2.destroyAllWindows()
# print(img)
