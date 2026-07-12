"""Roda um único tracker (mode + dataset) em processo próprio e grava as
previsões em JSON.

Usado pelo notebook de exemplos: instanciar múltiplos ClusWisard (Tr-WiSARD 1)
na mesma sessão Python contamina o estado global da extensão C++ `wisardpkg`
entre datasets. Rodar cada tracker em um processo isolado evita o problema —
é o mesmo motivo pelo qual `run.py` (um processo por execução) sempre produz
os números corretos.
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tr_wisard import TRWisard
from src.dataset import load_dataset, preload_frames, load_tuned_params


def _to_jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else value


def main():
    parser = argparse.ArgumentParser(description="Roda um tracker isolado e grava previsões em JSON")
    parser.add_argument("--mode", required=True, choices=list(TRWisard.MODES.keys()))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, help="Caminho do JSON de saída")
    parser.add_argument("--params-file", default=None,
                         help="JSON com os params a usar; se omitido, carrega de data/{dataset}/params-{mode}.json")
    args = parser.parse_args()

    image_paths, ground_truths = load_dataset(args.dataset)
    frames = preload_frames(image_paths)
    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
    else:
        params = load_tuned_params(args.mode, args.dataset) or TRWisard.default_params(args.mode)

    t0 = time.time()
    predictions = TRWisard(args.mode, frames, ground_truths, params=params).run()
    elapsed = time.time() - t0

    with open(args.out, "w") as f:
        json.dump({
            "predictions": [[_to_jsonable(v) for v in bbox] for bbox in predictions],
            "elapsed": elapsed,
        }, f)


if __name__ == "__main__":
    main()
