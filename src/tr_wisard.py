from src.trackers.tr_wisard1 import TrWisard1Tracker
from src.trackers.tr_wisard2 import TrWisard2Tracker


class TRWisard:

    MODES = {
        "tr_wisard1": TrWisard1Tracker,
        "tr_wisard2": TrWisard2Tracker,
    }

    def __init__(self, mode, frames, ground_truths, params=None):
        if mode not in self.MODES:
            raise ValueError(f"Modo inválido: '{mode}'. Opções: {list(self.MODES.keys())}")
        cls = self.MODES[mode]
        self._tracker = cls(params or cls.DEFAULT_PARAMS.copy(), frames, ground_truths)

    def run(self):
        return self._tracker.run()

    @classmethod
    def default_params(cls, mode):
        return cls.MODES[mode].DEFAULT_PARAMS.copy()

    @classmethod
    def default_grid(cls, mode):
        return cls.MODES[mode].DEFAULT_GRID.copy()
