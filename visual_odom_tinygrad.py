import numpy as np
from tinygrad import Tensor
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

def gaussian_filter(image):
    image = image.unsqueeze(0).unsqueeze(0)
    gaus = Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).unsqueeze(0).unsqueeze(0) / 16 
    gaus = image.conv2d(gaus, padding=1)
    return gaus.squeeze(0).squeeze(0)

def detect_corners(image, k=0.04, threshold=0.01):

    Ix, Iy = sobel_filter(image)

    Ixx = gaussian_filter(Ix * Ix)
    Iyy = gaussian_filter(Iy * Iy)
    Ixy = gaussian_filter(Ix * Iy)


    det = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    R = det - k * (trace ** 2)

    threshold = threshold * R.max()

    corners = R > threshold

    return corners

dataset = KITTIDataset(root_dir, mode="train")

img = Tensor(np.array(dataset[0]["left_img"]))
img = to_grayscale(img)

corners = detect_corners(image=img,threshold=0.15).numpy()
plt.imshow(corners, cmap='gray')
plt.show()
# cv2.imshow("img_gray", corners)


# cv2.waitKey(0) 
  
# closing all open windows 
# cv2.destroyAllWindows() 
# print(img)