import numpy as np
import cv2

# ArcFace reference 5-landmarks
ARC_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def distance2bbox(points, distance):
    return np.stack([
        points[:, 0] - distance[:, 0],
        points[:, 1] - distance[:, 1],
        points[:, 0] + distance[:, 2],
        points[:, 1] + distance[:, 3]
    ], axis=-1)

def distance2kps(points, distance):
    # distance shape = (N, 10)
    kps = []
    for i in range(0, 10, 2):
        x = points[:, 0] + distance[:, i]
        y = points[:, 1] + distance[:, i + 1]
        kps.append(np.stack([x, y], axis=-1))
    return np.stack(kps, axis=1)  # (N, 5, 2)

def align_face(img, kps):
    src = kps.astype(np.float32)
    dst = ARC_REF
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, (112, 112))
    return aligned
