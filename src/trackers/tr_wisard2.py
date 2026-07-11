import cv2
import numpy as np
from tqdm.auto import tqdm

from src.background import AdaptiveBackgroundModel


class WiSARDDiscriminator:

    def __init__(self, input_size, n_bits, seed):
        self.n_bits = n_bits
        self.n_rams = input_size // n_bits
        rng = np.random.RandomState(seed)
        perm = rng.permutation(input_size)
        self.mapping = perm[: self.n_rams * n_bits].reshape(self.n_rams, n_bits)
        self.rams = [set() for _ in range(self.n_rams)]
        self.retrain_count = 0
        self._powers = (1 << np.arange(n_bits - 1, -1, -1)).astype(np.int64)

    def _addresses(self, pattern):
        bits = pattern[self.mapping]
        return (bits.astype(np.int64) * self._powers).sum(axis=1)

    def train(self, pattern):
        for i, addr in enumerate(self._addresses(pattern)):
            self.rams[i].add(int(addr))

    def classify(self, pattern):
        addrs = self._addresses(pattern)
        activated = sum(int(a) in self.rams[i] for i, a in enumerate(addrs))
        return activated / self.n_rams


class TrWisard2Tracker:

    DEFAULT_PARAMS = {
        "WISARD_ADDRESS_SIZE": 3,
        "LIMIAR_RETREINO": 0.6,
        "LIMIAR_NOVO_DISC": 0.2,
        "QUEUE_MAX_SIZE": 10,
        "MAX_RETRAINS": 1,
        "SEARCH_RADIUS": 30,
        "STEP_SIZE": 3,
        "BACKGROUND_ALPHA": 0.5,
        "REMOVE_BACKGROUND": True,
        "SEED": 21,
    }

    DEFAULT_GRID = {
        "WISARD_ADDRESS_SIZE": [3, 5, 7],
        "LIMIAR_RETREINO": [0.8],
        "LIMIAR_NOVO_DISC": [0.4, 0.6],
        "QUEUE_MAX_SIZE": [5, 10, 25],
        "MAX_RETRAINS": [1, 3, 5],
        "SEARCH_RADIUS": [8, 10, 20],
        "STEP_SIZE": [3, 5],
        "BACKGROUND_ALPHA": [0.3, 0.5, 1.0],
        "REMOVE_BACKGROUND": [True, False],
        "SEED": [21],
    }

    def __init__(self, params, frames, ground_truths):
        self.params = params
        self.frames = frames
        self.ground_truths = ground_truths

    def _preprocess(self, frame, bg_model, prev_bbox=None):
        frame_bg = bg_model.apply(frame, prev_bbox) if bg_model is not None else frame.copy()
        if len(frame_bg.shape) == 3:
            return cv2.cvtColor(frame_bg, cv2.COLOR_RGB2GRAY)
        return frame_bg

    def _binarize_patch(self, patch_gray):
        mean = np.mean(patch_gray)
        return (patch_gray >= mean).astype(np.uint8)

    def _search_regions(self, prev_bbox, frame_shape):
        p = self.params
        x, y, w, h = prev_bbox
        cx, cy = x + w // 2, y + h // 2
        seen = set()
        for dx in range(-p["SEARCH_RADIUS"], p["SEARCH_RADIUS"] + 1, p["STEP_SIZE"]):
            for dy in range(-p["SEARCH_RADIUS"], p["SEARCH_RADIUS"] + 1, p["STEP_SIZE"]):
                nx = int(np.clip(cx + dx - w // 2, 0, frame_shape[1] - w))
                ny = int(np.clip(cy + dy - h // 2, 0, frame_shape[0] - h))
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    yield (nx, ny, w, h)

    def run(self):
        p = self.params
        frames = self.frames
        ground_truths = self.ground_truths

        first_gt = ground_truths[0]
        x0, y0, w0, h0 = map(int, first_gt)

        bg_model = None
        if p["REMOVE_BACKGROUND"]:
            bg_model = AdaptiveBackgroundModel(alpha=p["BACKGROUND_ALPHA"], threshold=5, bg_fill=100)
            bg_model.initialize(frames[0], first_gt)

        gray_first = self._preprocess(frames[0], bg_model, first_gt)
        patch0 = gray_first[y0:y0+h0, x0:x0+w0]
        pattern0 = self._binarize_patch(patch0).ravel()
        input_size = len(pattern0)

        first_disc = WiSARDDiscriminator(input_size, p["WISARD_ADDRESS_SIZE"], p["SEED"])
        first_disc.train(pattern0)
        disc_queue = [first_disc]

        prev_bbox = first_gt
        predictions = [prev_bbox]

        for i in tqdm(range(1, len(frames)), desc="TrWisard2"):
            gray = self._preprocess(frames[i], bg_model, prev_bbox)

            best_bbox = prev_bbox
            best_score = -1.0
            best_pattern = None
            best_disc_idx = 0

            for region in self._search_regions(prev_bbox, gray.shape):
                rx, ry, rw, rh = map(int, region)
                patch = gray[ry:ry+rh, rx:rx+rw]
                if patch.size == 0 or patch.shape[0] != h0 or patch.shape[1] != w0:
                    continue
                pattern = self._binarize_patch(patch).ravel()
                for di, disc in enumerate(disc_queue):
                    score = disc.classify(pattern)
                    if score > best_score:
                        best_score = score
                        best_bbox = region
                        best_pattern = pattern
                        best_disc_idx = di

            prev_bbox = best_bbox
            predictions.append(best_bbox)

            if best_score >= p["LIMIAR_RETREINO"]:
                pass
            elif best_score >= p["LIMIAR_NOVO_DISC"]:
                disc = disc_queue[best_disc_idx]
                if disc.retrain_count < p["MAX_RETRAINS"]:
                    disc.train(best_pattern)
                    disc.retrain_count += 1
            else:
                if len(disc_queue) < p["QUEUE_MAX_SIZE"]:
                    new_disc = WiSARDDiscriminator(input_size, p["WISARD_ADDRESS_SIZE"], p["SEED"] + len(disc_queue))
                    new_disc.train(best_pattern)
                    disc_queue.append(new_disc)
                else:
                    disc = disc_queue[0]
                    if disc.retrain_count < p["MAX_RETRAINS"]:
                        disc.train(best_pattern)
                        disc.retrain_count += 1

        return predictions
