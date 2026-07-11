import cv2
import numpy as np


class AdaptiveBackgroundModel:

    def __init__(self, alpha=0.01, threshold=25, bg_fill=0):
        self.alpha = alpha
        self.threshold = threshold
        self.bg_fill = bg_fill
        self.background = None

    def initialize(self, frame, bbox=None):
        bg = frame.astype(np.float32)
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            bg[y:y+h, x:x+w] = self.bg_fill
        self.background = bg

    def subtract(self, frame):
        """Returns binary foreground mask (255=foreground, 0=background)."""
        frame_f = frame.astype(np.float32)
        diff = cv2.absdiff(frame_f, self.background)
        gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return cv2.medianBlur(mask, 5)

    def update_with_mask(self, frame, learn_mask):
        """Update background only where learn_mask > 0."""
        frame_f = frame.astype(np.float32)
        mask_bool = (learn_mask > 0)
        self.background[mask_bool] = (
            (1 - self.alpha) * self.background[mask_bool]
            + self.alpha * frame_f[mask_bool]
        )

    def apply(self, frame, bbox=None):
        """Combined subtract+update (bg_fill-aware).

        If bbox is given, that region is forced to foreground so the tracked
        object is never absorbed into the background model.
        Background is updated everywhere EXCEPT the bbox region (matching
        the approach from commit 456fe20 that gave good results).
        """
        mask = self.subtract(frame)

        if bbox is not None:
            x, y, w, h = map(int, bbox)
            mask[y:y+h, x:x+w] = 255  # keep object pixels as foreground

        if self.bg_fill == 0:
            fg = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            fg = np.where(mask[:, :, None] > 0, frame, self.bg_fill).astype(np.uint8)

        # update background everywhere except bbox (all pixels except object region)
        learn_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            learn_mask[y:y+h, x:x+w] = 0
        self.update_with_mask(frame, learn_mask)

        return fg
