import cv2
import numpy as np
from collections import deque
from tqdm.auto import tqdm
from wisardpkg import ClusWisard

from src.background import AdaptiveBackgroundModel


class TrWisard1Tracker:

    DEFAULT_PARAMS = {
        "CLUSWISARD_ADDRESS_SIZE": 7,
        "CLUSWISARD_MIN_SCORE": 0.15,
        "STEP_SIZE": 5,
        "MAX_SEARCH_WINDOW_SCALE": 0.20,
        "BACKGROUND_ALPHA": 0,
        "CLUSWISARD_THRESHOLD": 1,
        "CLUSWISARD_DISCRIMINATOR_LIMIT": 5,
        "CLUSWISARD_BLEACHING_ACTIVATED": True,
        "CLUSWISARD_ACTIVATION_DEGREE": True,
        "CLUSWISARD_RETURN_CONFIDENCE": True,
        "CLUSWISARD_CLASSES_DEGREES": True,
        "REMOVE_BACKGROUND": False,
        "ANCHOR_SCORE": 0.05,
        "PASS_SCORE": 0.90,
        "SEED": 21,
    }

    DEFAULT_GRID = {
        "CLUSWISARD_ADDRESS_SIZE": [3, 5, 7],
        "CLUSWISARD_MIN_SCORE": [0.7, 0.8, 0.35],
        "STEP_SIZE": [3, 5],
        "MAX_SEARCH_WINDOW_SCALE": [0.5],
        "BACKGROUND_ALPHA": [0.05, 0.1, 1],
        "ANCHOR_SCORE": [0.6, 0.7, 0.8],
        "PASS_SCORE": [0.8, 0.9],
        "CLUSWISARD_DISCRIMINATOR_LIMIT": [3, 5, 10],
        "REMOVE_BACKGROUND": [True, False],
    }

    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, params, frames, ground_truths):
        self.params = params
        self.frames = frames
        self.ground_truths = ground_truths

    def _new_clus(self):
        p = self.params
        return ClusWisard(
            p["CLUSWISARD_ADDRESS_SIZE"],
            p["CLUSWISARD_MIN_SCORE"],
            p["CLUSWISARD_THRESHOLD"],
            p["CLUSWISARD_DISCRIMINATOR_LIMIT"],
            bleachingActivated=p["CLUSWISARD_BLEACHING_ACTIVATED"],
            returnActivationDegree=p["CLUSWISARD_ACTIVATION_DEGREE"],
            returnConfidence=p["CLUSWISARD_RETURN_CONFIDENCE"],
            returnClassesDegrees=p["CLUSWISARD_CLASSES_DEGREES"],
            seed=21,
        )

    def _preprocess(self, frame, bg_model, bbox=None):
        p = self.params
        if bg_model is None:
            frame_bg = frame
        else:
            frame_bg = bg_model.apply(frame, bbox)
        gray = cv2.cvtColor(frame_bg, cv2.COLOR_RGB2GRAY) if len(frame_bg.shape) == 3 else frame_bg.copy()
        gray = self._CLAHE.apply(gray)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (otsu > 0).astype(np.uint8)

    def _search_regions(self, prev_bbox, frame_shape):
        p = self.params
        x, y, w, h = prev_bbox
        cx, cy = x + w // 2, y + h // 2
        radius = int(max(w, h) * p["MAX_SEARCH_WINDOW_SCALE"])
        for dx in range(-radius, radius, p["STEP_SIZE"]):
            for dy in range(-radius, radius, p["STEP_SIZE"]):
                nx = int(np.clip(cx + dx - w // 2, 0, frame_shape[1] - w))
                ny = int(np.clip(cy + dy - h // 2, 0, frame_shape[0] - h))
                yield (nx, ny, w, h)

    def run(self):
        p = self.params
        frames = self.frames
        ground_truths = self.ground_truths

        bg_model = None
        if p["REMOVE_BACKGROUND"]:
            bg_model = AdaptiveBackgroundModel(alpha=p["BACKGROUND_ALPHA"], threshold=25, bg_fill=0)
            bg_model.initialize(frames[0])

        x0, y0, w0, h0 = map(int, ground_truths[0])
        bin_first = self._preprocess(frames[0], bg_model, ground_truths[0])
        first_pattern = bin_first[y0:y0+h0, x0:x0+w0].flatten()

        anchor_patterns = [first_pattern.tolist()]
        clus = self._new_clus()
        clus.train(anchor_patterns, ["object"])

        discriminator_queue = deque(maxlen=p["CLUSWISARD_DISCRIMINATOR_LIMIT"])
        prev_bbox = ground_truths[0]
        predictions = [prev_bbox]

        for i in tqdm(range(1, len(frames)), desc="TrWisard1"):
            bin_frame = self._preprocess(frames[i], bg_model, prev_bbox)

            best_bbox = prev_bbox
            best_score = -1
            best_patch = None

            for region in self._search_regions(prev_bbox, bin_frame.shape):
                x, y, w, h = map(int, region)
                patch = bin_frame[y:y+h, x:x+w]
                if patch.size == 0:
                    continue
                pattern = patch.ravel()
                result = clus.classify([pattern])[0]
                score = result.get("activationDegree", 0)
                if score > best_score:
                    best_score = score
                    best_bbox = region
                    best_patch = pattern
                if score > p["PASS_SCORE"]:
                    break

            prev_bbox = best_bbox
            predictions.append(best_bbox)

            if best_score <= p["CLUSWISARD_MIN_SCORE"]:
                if best_patch is not None:
                    discriminator_queue.append({"patch": best_patch.tolist(), "activation": best_score})

                if len(discriminator_queue) >= p["CLUSWISARD_DISCRIMINATOR_LIMIT"]:
                    best_disc = max(discriminator_queue, key=lambda d: d["activation"])
                    if best_disc["activation"] >= p["ANCHOR_SCORE"]:
                        anchor_patterns.append(best_disc["patch"])
                    discriminator_queue.clear()

                clus = self._new_clus()
                X_train = anchor_patterns.copy()
                y_train = ["object"] * len(anchor_patterns)
                for disc in discriminator_queue:
                    X_train.append(disc["patch"])
                    y_train.append("object")
                clus.train(X_train, y_train)

        return predictions
