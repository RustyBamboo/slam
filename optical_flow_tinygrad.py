import numpy as np
from tinygrad import Tensor, dtypes
import os

import cv2
from PIL import Image

import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"

root_dir = "/mnt/storage/dataset/sequences/00"
pose_path = "/mnt/storage/dataset/poses/00.txt"




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


def to_grayscale(image):
    gray = Tensor([0.299, 0.587, 0.114])
    return Tensor.einsum("abc,c->ab", image, gray) / 255

def convolve(image, f):
    image = image.unsqueeze(0).unsqueeze(0)
    f = f.unsqueeze(0).unsqueeze(0)
    image = image.conv2d(f, stride=1, padding=1)
    return image.squeeze(0).squeeze(0)

def sobel_filter(image):
    """
    Apply Sobel filter to the input image. This provides a good approximation of the gradient.
    """
    image = image.unsqueeze(0).unsqueeze(0)
    Kx = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
    Ky = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
    Ix = image.conv2d(Kx, padding=1)
    Iy = image.conv2d(Ky, padding=1)
    return Ix.squeeze(0).squeeze(0), Iy.squeeze(0).squeeze(0)



def horn_schunck(image1, image2, num_iter=10, alpha=0.4):
    horizontal_flow = Tensor.zeros_like(image1)
    vertical_flow = Tensor.zeros_like(image2)

    # kernel_x = Tensor([[-1, 1], [-1, 1]]) * 0.25
    # kernel_y = Tensor([[-1, -1], [1, 1]]) * 0.25
    # kernel_t = Tensor([[1, 1], [1, 1]]) * 0.25
    kernel_laplacian = Tensor(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]]
    )

    fx1, fy1 = sobel_filter(image1)
    fx2, fy2 = sobel_filter(image2)

    fx = fx1 + fx2
    fy = fy1 + fy2
    ft = image2 - image1
    # ft = -fx1 + fx2 - fy1 + fy2
    # fx = convolve(image1, kernel_x) + convolve(image2, kernel_x)
    # fy = convolve(image1, kernel_y) + convolve(image2, kernel_y)
    # ft = convolve(image1, -kernel_t) + convolve(image2, kernel_t)

    for _ in range(num_iter):
        horizontal_flow_avg = convolve(horizontal_flow, kernel_laplacian)
        vertical_flow_avg = convolve(vertical_flow, kernel_laplacian)

        p = fx * horizontal_flow_avg + fy * vertical_flow_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2

        horizontal_flow = horizontal_flow_avg - fx * (p / d)
        vertical_flow = vertical_flow_avg - fy * (p / d)

    return horizontal_flow, vertical_flow


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