import os

from PIL import Image

import numpy as np

import cv2


os.environ["QT_QPA_PLATFORM"] = "xcb"


root_dir = "/mnt/storage/dataset/sequences/00"
pose_path = "/mnt/storage/dataset/poses/00.txt"

with open(pose_path) as f:
    pose = f.readlines()

pose_position = []
for p in pose:
    p = p.strip().split()
    x = float(p[3])
    y = float(p[7])
    z = float(p[11])
    pose_position.append(np.array([[x], [y], [z]]))


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

# Calibration Matrix for KITTI sample
k_cam = np.array(
    [
        [7.188560000000e02, 0.000000000000e00, 6.071928000000e02],
        [0.000000000000e00, 7.188560000000e02, 1.852157000000e02],
        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00],
    ]
)


class MonoVisualOdom(object):
    def __init__(
        self,
        dataset,
        focal_length=718.8560,
        pp=(607.1928, 185.2157),
        lk_params=dict(
            winSize=(31, 31),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
        ),
        detector=cv2.FastFeatureDetector_create(threshold=100, nonmaxSuppression=True),
    ):
        self.dataset = dataset
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3), dtype=np.float32)
        self.t = np.zeros(shape=(3, 3), dtype=np.float32)
        self.id = 0
        self.n_features = 0
        self.sc = 1

        self.trajectory = []

        self.process_frame()

    def detect(self, img):
        p0 = self.detector.detect(img)
        self.key_points = p0
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometry(self):
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame, self.current_frame, self.p0, None, **self.lk_params
        )
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        if self.id < 2:
            E, _ = cv2.findEssentialMat(
                self.good_new,
                self.good_old,
                self.focal,
                self.pp,
                cv2.RANSAC,
                0.999,
                1.0,
                None,
            )
            _, self.R, self.t, _ = cv2.recoverPose(
                E,
                self.good_old,
                self.good_new,
                self.R,
                self.t,
                focal=self.focal,
                pp=self.pp,
                # None,
            )
        else:
            E, _ = cv2.findEssentialMat(
                self.good_new,
                self.good_old,
                self.focal,
                self.pp,
                cv2.RANSAC,
                0.999,
                1.0,
                None,
            )
            _, R, t, _ = cv2.recoverPose(
                E,
                self.good_old,
                self.good_new,
                self.R.copy(),
                self.t.copy(),
                focal=self.focal,
                pp=self.pp,
                # None,
            )

            absolute_scale = self.sc
            if (
                absolute_scale > 0.1
                and abs(t[2][0]) > abs(t[0][0])
                and abs(t[2][0]) > abs(t[1][0])
            ):
                self.t = self.t + absolute_scale * self.R.dot(t)
                self.R = R.dot(self.R)

        self.n_features = self.good_new.shape[0]

    def get_mono_coordinates(self):
        diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def process_frame(self):
        if self.id < 2:
            self.old_frame = np.array(self.dataset[0]["left_img"])
            self.current_frame = np.array(self.dataset[1]["left_img"])
            self.visual_odometry()
            self.id = 2

            r_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            t_init = np.array([[0], [0], [0]])
            self.n_cloud = self.triangulaion(
                r_init, t_init, self.good_old, self.good_new
            )
            self.sc = 1
        else:
            self.old_frame = self.current_frame
            self.current_frame = np.array(self.dataset[self.id]["left_img"])
            self.visual_odometry()
            self.id += 1

            o_cloud = self.n_cloud.copy()
            self.n_cloud = self.triangulaion(
                self.R, self.t, self.good_old, self.good_new
            )
            self.sc = self.scale(o_cloud, self.n_cloud)

            self.trajectory.append(np.array([self.t[0], self.t[2], self.t[0]]))

    def triangulaion(self, R, t, pt1, pt2):
        R = self.R
        t = self.t
        pr = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        pr_mat = np.dot(k_cam, pr)
        P = np.hstack((R, t))
        P1 = np.dot(k_cam, P)
        # making matrix 2xn :
        ch1 = pt1.transpose()
        ch2 = pt2.transpose()
        cloud = cv2.triangulatePoints(
            projMatr1=pr_mat, projMatr2=P1, projPoints1=ch1, projPoints2=ch2
        )
        cloud = cloud[:4, :]
        return cloud

    def scale(self, O_cloud, N_cloud):
        siz = min(O_cloud.shape[1], N_cloud.shape[1])
        o_c = np.zeros((3, siz))
        n_c = np.zeros((3, siz))
        o_c = O_cloud[:, :siz]
        n_c = N_cloud[:, :siz]
        # axis one and shift = 1 == rolling column by one
        o_c1 = np.roll(o_c, axis=1, shift=1)
        n_c1 = np.roll(n_c, axis=1, shift=1)
        # axis = 0 == taking norm along column
        # print(np.sum(np.sign(n_c - n_c1)))
        scale = np.linalg.norm((o_c - o_c1), axis=0) / (
            np.linalg.norm(n_c - n_c1, axis=0) + 1e-8
        )
        # taking median along the row (norm)
        scale = np.median(scale)
        return scale


focal = 718.8560
pp = (607.1928, 185.2157)
R_total = np.zeros((3, 3))
t_total = np.empty(shape=(3, 1))

odom = MonoVisualOdom(dataset, focal, pp)

traj = np.zeros(shape=(600, 800, 3))

for t in pose_position:
    cv2.circle(traj, (int(t[0]) + 400, int(t[2]) + 100), 1, list((255, 0, 0)), 4)


for img_id in range(4539):
    frame = odom.current_frame

    for point in odom.good_old:
        cv2.circle(frame, center=tuple(point.astype(int)), radius=1, color=(255, 0, 0))

    for point in odom.good_new:
        cv2.circle(frame, center=tuple(point.astype(int)), radius=1, color=(0, 0, 255))

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)

    odom.process_frame()

    mono_coord = odom.get_mono_coordinates()

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]

    # traj = cv.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
    traj = cv2.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

    cv2.imshow("trajectory", traj)

    if k == ord("q"):
        break

cv2.imwrite("map.png", traj)

cv2.destroyAllWindows()
