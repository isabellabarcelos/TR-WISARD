import glob
import json
import time
import cv2
from pathlib import Path

from src.utils.tracker_utils import load_ground_truth_from_gt_txt

DATA_ROOT = Path(__file__).parent.parent / "data"


def load_dataset(dataset, data_root=None):
    root = Path(data_root) if data_root else DATA_ROOT
    dataset_root = root / dataset
    image_paths = sorted(glob.glob(str(dataset_root / "imgs" / "*.png")))
    ground_truths = load_ground_truth_from_gt_txt(str(dataset_root / f"{dataset}_gt.txt"))
    return image_paths, ground_truths


def load_tuned_params(mode, dataset, data_root=None):
    root = Path(data_root) if data_root else DATA_ROOT
    params_path = root / dataset / f"params-{mode.replace('_', '-')}.json"
    if not params_path.exists():
        return None
    with open(params_path) as f:
        return json.load(f)


def preload_frames(image_paths):
    print(f"Pré-carregando {len(image_paths)} frames...")
    t0 = time.time()
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]
    print(f"  Done em {time.time() - t0:.1f}s")
    return frames
