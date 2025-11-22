import cv2
import numpy as np
import onnxruntime as ort

# ============================================================
# SCRFD HELPERS
# ============================================================

def distance2bbox(points, distance):
    return np.stack([
        points[:, 0] - distance[:, 0],
        points[:, 1] - distance[:, 1],
        points[:, 0] + distance[:, 2],
        points[:, 1] + distance[:, 3]
    ], axis=-1)

def distance2kps(points, distance):
    """ Convert SCRFD 10-keypoint distance to 5 landmark pairs """
    # distance shape = (N, 10)
    kps = []
    for i in range(0, 10, 2):
        x = points[:, 0] + distance[:, i]
        y = points[:, 1] + distance[:, i + 1]
        kps.append(np.stack([x, y], axis=-1))
    return np.stack(kps, axis=1)  # (N, 5, 2)

# ============================================================
# ALIGN FACE FOR ARCFACE
# ============================================================

# ArcFace reference 5-landmarks
ARC_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, kps):
    src = kps.astype(np.float32)
    dst = ARC_REF

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, (112, 112))
    return aligned

# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline(scrfd_path, arcface_path, image_path):

    print("\n===== LOADING SCRFD MODEL =====")
    det_sess = ort.InferenceSession(scrfd_path, providers=["CPUExecutionProvider"])
    det_input = det_sess.get_inputs()[0].name

    img = cv2.imread(image_path)
    ih, iw = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1/128.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True
    )

    outs = det_sess.run(None, {det_input: blob})

    cls_8, cls_16, cls_32 = outs[0], outs[1], outs[2]
    box_8, box_16, box_32 = outs[3], outs[4], outs[5]
    kps_8, kps_16, kps_32 = outs[6], outs[7], outs[8]

    # ============================================================
    # Generate center grids for each FPN layer
    # ============================================================

    def make_centers(h, w, stride):
        y, x = np.mgrid[:h, :w]
        pts = np.stack([x, y], axis=-1).reshape(-1, 2)
        pts = pts * stride
        return np.repeat(pts, 2, axis=0)   # 2 anchors per location

    centers_8  = make_centers(80, 80, 8)
    centers_16 = make_centers(40, 40, 16)
    centers_32 = make_centers(20, 20, 32)

    # ============================================================
    # Decode heads separately, then concatenate (VALID)
    # ============================================================

    scores = np.concatenate([
        cls_8.reshape(-1),
        cls_16.reshape(-1),
        cls_32.reshape(-1)
    ])

    boxes = np.concatenate([
        distance2bbox(centers_8,  box_8),
        distance2bbox(centers_16, box_16),
        distance2bbox(centers_32, box_32)
    ])

    kpss = np.concatenate([
        distance2kps(centers_8,  kps_8),
        distance2kps(centers_16, kps_16),
        distance2kps(centers_32, kps_32)
    ])

    # ============================================================
    # Select best detected face
    # ============================================================

    idx = np.argmax(scores)
    conf = scores[idx]

    if conf < 0.4:
        print("❌ No high-confidence face found")
        return

    box = boxes[idx].astype(int)
    kps = kpss[idx]

    print("\n===== DETECTION RESULT =====")
    print("Confidence:", conf)
    print("Box:", box)
    print("Landmarks:\n", kps)

    # ============================================================
    # Align face for ArcFace
    # ============================================================

    aligned = align_face(img, kps)

    # ============================================================
    # ArcFace Embedding
    # ============================================================

    print("\n===== RUNNING ARCFACE =====")
    arc = ort.InferenceSession(arcface_path, providers=["CPUExecutionProvider"])
    arc_input = arc.get_inputs()[0].name

    face = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))[None, :]

    emb = arc.run(None, {arc_input: face})[0][0]

    print("Embedding length:", len(emb))
    print("Embedding sample:", emb[:10])


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    SCRFD = "det_2.5g.onnx"
    ARCFACE = "w600k_r50.onnx"
    IMG = "WIN_20240821_10_35_15_Pro.jpg"

    run_pipeline(SCRFD, ARCFACE, IMG)
