import numpy as np
import cv2

# For SCRFD
def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.vstack((x1, y1, x2, y2)).T


def distance2kps(points, distance):
    num_kps = distance.shape[1] // 2
    kps = []
    for i in range(num_kps):
        x = points[:, 0] + distance[:, i * 2]
        y = points[:, 1] + distance[:, i * 2 + 1]
        kps.append(np.vstack((x, y)).T)
    return np.stack(kps, axis=1)


# ArcFace template
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(img, kps, size=(112, 112)):
    kps = kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(kps, arcface_dst)
    aligned = cv2.warpAffine(img, M, size)
    return aligned
