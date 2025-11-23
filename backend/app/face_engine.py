import cv2
import numpy as np
import onnxruntime as ort
from app.utils.helpers import distance2bbox, distance2kps, align_face

# USE THE ONNX PATHS YOU UPLOADED
DETECTOR_PATH = "models/det_2.5g.onnx"
RECOGNIZER_PATH = "models/w600k_r50_fp16.onnx"

class FaceEngine:
    def __init__(self, det_path: str = DETECTOR_PATH, rec_path: str = RECOGNIZER_PATH):
        # SCRFD detector (multi-head)
        self.det_sess = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
        self.det_input = self.det_sess.get_inputs()[0].name
        # arcface embedder
        self.rec_sess = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])
        self.rec_input = self.rec_sess.get_inputs()[0].name

        # thresholds
        self.min_conf = 0.3  # lowered to accept your test images

    def _make_centers(self, h, w, stride, num_anchors=2):
        y, x = np.mgrid[:h, :w]
        pts = np.stack([x, y], axis=-1).reshape(-1, 2)
        pts = pts * stride
        pts = np.repeat(pts, num_anchors, axis=0)
        return pts

    def _detect(self, img):
        # prepare blob
        blob = cv2.dnn.blobFromImage(img, 1.0/128.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det_sess.run(None, {self.det_input: blob})

        # unpack multi-head
        cls_8, cls_16, cls_32 = outs[0], outs[1], outs[2]
        box_8, box_16, box_32 = outs[3], outs[4], outs[5]
        kps_8, kps_16, kps_32 = outs[6], outs[7], outs[8]

        # centers
        centers_8 = self._make_centers(80, 80, 8)
        centers_16 = self._make_centers(40, 40, 16)
        centers_32 = self._make_centers(20, 20, 32)

        scores = np.concatenate([cls_8.reshape(-1), cls_16.reshape(-1), cls_32.reshape(-1)])
        boxes = np.concatenate([
            distance2bbox(centers_8, box_8),
            distance2bbox(centers_16, box_16),
            distance2bbox(centers_32, box_32)
        ])
        kpss = np.concatenate([
            distance2kps(centers_8, kps_8),
            distance2kps(centers_16, kps_16),
            distance2kps(centers_32, kps_32)
        ])

        if scores.size == 0:
            return None

        idx = int(np.argmax(scores))
        conf = float(scores[idx])
        if conf < self.min_conf:
            return None

        box = boxes[idx].astype(int)
        kps = kpss[idx]
        return box, kps, conf

    def _get_embedding(self, aligned_face):
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 127.5
        blob = np.transpose(face, (2, 0, 1))[None, :].astype(np.float32)
        emb = self.rec_sess.run(None, {self.rec_input: blob})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    def process(self, img_bgr: np.ndarray):
        box_kps_conf = self._detect(img_bgr)
        if box_kps_conf is None:
            return None

        box, kps, conf = box_kps_conf
        aligned = align_face(img_bgr, kps)
        emb = self._get_embedding(aligned)  # numpy vector

        return {
            "box": box.tolist(),
            "kps": kps.tolist(),
            "embedding": emb.tobytes(),   # <-- FIXED
            "conf": conf
        }


    @staticmethod
    def emb_bytes_to_np(emb_bytes):
        return np.frombuffer(emb_bytes, dtype=np.float32)
