import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tr_wisard import TRWisard
from src.metrics import compute_errors, save_results, create_experiment_folder, get_top_n_experiments
from src.dataset import load_dataset, preload_frames, load_tuned_params


def run_single(mode, params, frames, ground_truths, output_root, image_paths):
    exp_path = create_experiment_folder(output_root)

    with open(exp_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    t0 = time.time()
    predictions = TRWisard(mode, frames, ground_truths, params=params).run()
    exec_time = time.time() - t0

    frame_idxs, errors = compute_errors(predictions, ground_truths)
    save_results(predictions, frame_idxs, errors, exec_time, exp_path, image_paths=image_paths)

    print(f"\nErro médio:  {np.mean(errors):.2f} px")
    print(f"Erro máximo: {np.max(errors):.2f} px")
    print(f"FPS:         {len(predictions) / exec_time:.1f}")
    print(f"Pasta:       {exp_path}")


def run_grid(mode, frames, ground_truths, output_root, start_from=0):
    grid = TRWisard.default_grid(mode)
    params_base = TRWisard.default_params(mode)
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    total = len(combinations)

    print(f"[Grid] {total} combinações | start_from={start_from}")

    best_error = float("inf")
    best_exp = None
    best_params = None

    for idx, values in enumerate(combinations):
        if idx < start_from:
            continue

        params = params_base.copy()
        for k, v in zip(keys, values):
            params[k] = v

        print(f"\n[{idx+1}/{total}] {dict(zip(keys, values))}")

        exp_path = create_experiment_folder(output_root)
        with open(exp_path / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        try:
            t0 = time.time()
            predictions = TRWisard(mode, frames, ground_truths, params=params).run()
            exec_time = time.time() - t0

            frame_idxs, errors = compute_errors(predictions, ground_truths)
            mean_err = float(np.mean(errors))
            save_results(predictions, frame_idxs, errors, exec_time, exp_path,
                         save_video=False, save_plot=False)

            print(f"  Erro médio: {mean_err:.2f} px  |  FPS: {len(predictions)/exec_time:.1f}")

            if mean_err < best_error:
                best_error = mean_err
                best_exp = exp_path
                best_params = params.copy()

        except Exception as e:
            print(f"  ERRO: {e}")

    print("\n=== MELHOR RESULTADO ===")
    print(f"Erro médio: {best_error:.2f} px")
    print(f"Pasta:      {best_exp}")
    if best_params:
        print(f"Params:     {json.dumps(best_params, indent=2)}")


def run_top(mode, dataset, output_root, n=5):
    print(f"\nTop {n} experimentos — {mode} / {dataset}:")
    for exp_id, err, p in get_top_n_experiments(output_root, n=n):
        print(f"  #{exp_id}  erro={err:.4f}  |  {p}")


def main():
    parser = argparse.ArgumentParser(description="TR-WISARD Object Tracker")
    parser.add_argument("--mode", required=True, choices=list(TRWisard.MODES.keys()),
                        help="Metodologia: cluswisard | wisard_discriminator")
    parser.add_argument("--dataset", required=True,
                        help="Nome do dataset (ex: tiger1, sylv, david)")
    parser.add_argument("--run", default="single", choices=["single", "grid", "top"],
                        help="Modo de execução")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Índice inicial para retomar grid search")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Quantos melhores experimentos exibir no modo top")
    parser.add_argument("--data-root", default=None,
                        help="Raiz dos datasets (padrão: ./data)")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "data" / "experimentos" / args.dataset / args.mode

    if args.run == "top":
        run_top(args.mode, args.dataset, output_root, n=args.top_n)
        return

    image_paths, ground_truths = load_dataset(args.dataset, args.data_root)
    if not image_paths:
        print(f"Nenhuma imagem encontrada para o dataset '{args.dataset}'")
        sys.exit(1)

    frames = preload_frames(image_paths)

    if args.run == "single":
        params = load_tuned_params(args.mode, args.dataset, args.data_root)
        if params is None:
            print(f"Nenhum params-{args.mode.replace('_', '-')}.json encontrado para '{args.dataset}', usando DEFAULT_PARAMS genérico")
            params = TRWisard.default_params(args.mode)
        run_single(args.mode, params, frames, ground_truths, output_root, image_paths)
    else:
        run_grid(args.mode, frames, ground_truths, output_root, start_from=args.start_from)


if __name__ == "__main__":
    main()
