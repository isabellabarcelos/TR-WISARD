import json
import re
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2
from pathlib import Path


def _fmt_dataset(name):
    """'tiger1' → 'Tiger 1', 'faceocc2' → 'Faceocc 2', 'dollar' → 'Coupon Book'."""
    _aliases = {
        "dollar":   "Coupon Book",
        "david":    "David Indoor",
        "faceocc":  "Occluded Face",
        "faceocc2": "Occluded Face 2",
        "sylv":     "Sylvester",
    }
    name = str(name)
    if name in _aliases:
        return _aliases[name]
    s = re.sub(r'([a-zA-Z])(\d+)$', r'\1 \2', name)
    return s[0].upper() + s[1:] if s else s


def _fmt_tracker(name):
    """'TrWisard1', 'tr_wisard1', etc. → 'Tr-WiSARD 1'."""
    m = re.search(r'(\d+)$', str(name))
    return f"Tr-WiSARD {m.group(1)}" if m else str(name)

# Cores compartilhadas entre vídeo (BGR) e gráficos (hex) — A = azul sólido, B = vermelho pontilhado
_COLOR_A_BGR = (255, 100, 30)
_COLOR_B_BGR = (30, 30, 220)
_COLOR_A_HEX = "#1e64ff"
_COLOR_B_HEX = "#dc1e1e"
_COLOR_GT_HEX = "#2ca02c"


# ── métricas por frame ──────────────────────────────────────────────────────

def _is_visible(gt):
    return gt != (0.0, 0.0, 0.0, 0.0) and gt[2] > 0 and gt[3] > 0


def compute_cle(predictions, ground_truths):
    """CLE (Central Location Error) por frame — ignora frames com gt invisível (0,0,0,0)."""
    frames, errors = [], []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if not _is_visible(gt):
            continue
        cx_gt = np.array([gt[0] + gt[2] / 2,    gt[1] + gt[3] / 2])
        cx_pr = np.array([pred[0] + pred[2] / 2, pred[1] + pred[3] / 2])
        frames.append(i)
        errors.append(float(np.linalg.norm(cx_gt - cx_pr)))
    return frames, errors


def compute_jaccard(predictions, ground_truths):
    """Coeficiente de Jaccard (IoU) em % por frame — ignora frames com gt invisível (0,0,0,0)."""
    frames, jaccards = [], []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if not _is_visible(gt):
            continue
        x1p, y1p = pred[0], pred[1]
        x2p, y2p = pred[0] + pred[2], pred[1] + pred[3]
        x1g, y1g = gt[0], gt[1]
        x2g, y2g = gt[0] + gt[2], gt[1] + gt[3]
        inter = max(0.0, min(x2p, x2g) - max(x1p, x1g)) * max(0.0, min(y2p, y2g) - max(y1p, y1g))
        union = pred[2] * pred[3] + gt[2] * gt[3] - inter
        frames.append(i)
        jaccards.append(inter / union * 100.0 if union > 0 else 0.0)
    return frames, jaccards


def compute_errors(predictions, ground_truths):
    """CLE amostrado a cada 5 frames, ignorando gt invisível — usado por run.py."""
    errors, frames = [], []
    for i in range(0, len(predictions), 5):
        gt, pred = ground_truths[i], predictions[i]
        if not _is_visible(gt):
            continue
        cx_gt = np.array([gt[0] + gt[2] / 2,   gt[1] + gt[3] / 2])
        cx_pr = np.array([pred[0] + pred[2] / 2, pred[1] + pred[3] / 2])
        errors.append(float(np.linalg.norm(cx_gt - cx_pr)))
        frames.append(i)
    return frames, errors


# ── comparação dois trackers ────────────────────────────────────────────────

def save_comparison(
    dataset_name, exp_path,
    preds_a, name_a, elapsed_a,
    preds_b, name_b, elapsed_b,
    ground_truths, image_paths=None, frames=None,
):
    exp_path = Path(exp_path)
    metrics_path = exp_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    frames_cle, errors_a   = compute_cle(preds_a, ground_truths)
    _,          errors_b   = compute_cle(preds_b, ground_truths)
    frames_jac, jaccards_a = compute_jaccard(preds_a, ground_truths)
    _,          jaccards_b = compute_jaccard(preds_b, ground_truths)
    fps_a = len(preds_a) / elapsed_a
    fps_b = len(preds_b) / elapsed_b

    # metrics.txt
    with open(metrics_path / "metrics.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Frames avaliados (CLE):    {len(frames_cle)}\n")
        f.write(f"Frames avaliados (Jaccard):{len(frames_jac)}\n\n")
        for name, errs, jacs, fps in [
            (name_a, errors_a, jaccards_a, fps_a),
            (name_b, errors_b, jaccards_b, fps_b),
        ]:
            f.write(f"[{name}]\n")
            f.write(f"Erro medio (CLE):  {np.mean(errs):.4f} px\n")
            f.write(f"Jaccard medio:     {np.mean(jacs):.4f} %\n")
            f.write(f"FPS:               {fps:.2f}\n\n")

    # erro.txt
    with open(metrics_path / "erro.txt", "w") as f:
        f.write(f"frame,{name_a},{name_b}\n")
        for i, f_idx in enumerate(frames_cle):
            f.write(f"{f_idx},{errors_a[i]:.4f},{errors_b[i]:.4f}\n")

    # jaccard.txt
    with open(metrics_path / "jaccard.txt", "w") as f:
        f.write(f"frame,{name_a},{name_b}\n")
        for i, f_idx in enumerate(frames_jac):
            f.write(f"{f_idx},{jaccards_a[i]:.4f},{jaccards_b[i]:.4f}\n")

    # gráficos
    _ds_slug = _fmt_dataset(dataset_name).lower().replace(" ", "_")
    _plot_comparison(
        dataset_name, frames_cle,
        errors_a, name_a, errors_b, name_b,
        "Erro de Localização Central por frame", "CLE (pixels)",
        metrics_path / f"cle_{_ds_slug}.png",
        ylim=(0, 100),
    )
    _plot_comparison(
        dataset_name, frames_jac,
        jaccards_a, name_a, jaccards_b, name_b,
        "Coeficiente de Jaccard por frame", "Coeficiente de Jaccard (%)",
        metrics_path / f"jaccard_{_ds_slug}.png",
        ylim=(0, 105),
    )

    # grid de exemplo (amostras com ground truth + as duas bboxes)
    source, from_memory = (frames, True) if frames is not None else (image_paths, False)
    if source:
        loader = _make_frame_loader(source, from_memory)
        n_src = min(len(source), len(ground_truths))
        _plot_example_grid(
            dataset_name, loader, n_src, ground_truths,
            preds_a, name_a, preds_b, name_b,
            metrics_path / "grid.png",
        )

    # vídeo com os dois bboxes
    if frames is not None:
        _save_comparison_video(preds_a, name_a, preds_b, name_b, frames, exp_path, from_memory=True)
    elif image_paths:
        _save_comparison_video(preds_a, name_a, preds_b, name_b, image_paths, exp_path, from_memory=False)

    return {
        name_a: {"mean_cle": np.mean(errors_a), "mean_jaccard": np.mean(jaccards_a), "fps": fps_a},
        name_b: {"mean_cle": np.mean(errors_b), "mean_jaccard": np.mean(jaccards_b), "fps": fps_b},
    }


def _plot_comparison(dataset_name, frames, vals_a, name_a, vals_b, name_b, title, ylabel, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, vals_a, "-",  color="#1f77b4", linewidth=1.8, label=_fmt_tracker(name_a))
    ax.plot(frames, vals_b, "--", color="#d62728", linewidth=1.8, label=_fmt_tracker(name_b))
    ax.set_title(f"{title} — {_fmt_dataset(dataset_name)}")
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_frame_loader(source, from_memory):
    """Retorna fn(i) -> frame RGB, sem carregar tudo em memória quando vier de disco."""
    if from_memory:
        return lambda i: source[i]
    return lambda i: cv2.cvtColor(cv2.imread(source[i]), cv2.COLOR_BGR2RGB)


def _plot_example_grid(
    dataset_name, frame_loader, n_frames, ground_truths,
    preds_a, name_a, preds_b, name_b, out_path,
    max_frames=20, cols=5,
):
    """Grid de amostras uniformemente distribuídas nos frames com gt visível."""
    valid_frames = [i for i in range(n_frames) if i < len(ground_truths) and _is_visible(ground_truths[i])]
    if not valid_frames:
        return

    n_valid = len(valid_frames)
    if n_valid <= max_frames:
        sampled = valid_frames
    else:
        idx = np.round(np.linspace(0, n_valid - 1, max_frames)).astype(int)
        sampled = [valid_frames[i] for i in idx]

    n_disp = len(sampled)
    rows = (n_disp + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.2, rows * 2.6))
    gs = fig.add_gridspec(rows, cols, wspace=0.02, hspace=0.18)

    for plot_idx, frame_idx in enumerate(sampled):
        r, c = divmod(plot_idx, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

        img = frame_loader(frame_idx)
        gray = img.mean(axis=2) if img.ndim == 3 else img
        ax.imshow(gray, cmap="gray", interpolation="nearest")

        if frame_idx < len(preds_a):
            x, y, w, h = preds_a[frame_idx]
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=_COLOR_A_HEX, facecolor="none"))
        if frame_idx < len(preds_b):
            x, y, w, h = preds_b[frame_idx]
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=_COLOR_B_HEX, facecolor="none", linestyle="--"))
        x, y, w, h = ground_truths[frame_idx]
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=_COLOR_GT_HEX, facecolor="none"))

        ax.set_title(f"F{frame_idx}", fontsize=7, pad=1)

    for k in range(n_disp, rows * cols):
        rr, cc = divmod(k, cols)
        fig.add_subplot(gs[rr, cc]).axis("off")

    legend_handles = [
        Line2D([0], [0], color=_COLOR_GT_HEX, linewidth=2, label="Ground Truth"),
        Line2D([0], [0], color=_COLOR_A_HEX, linewidth=2, label=_fmt_tracker(name_a)),
        Line2D([0], [0], color=_COLOR_B_HEX, linewidth=2, linestyle="--", label=_fmt_tracker(name_b)),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.09)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_dashed_line(frame, pt1, pt2, color, thickness=2, dash=10, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos, draw = 0, True
    while pos < length:
        end = min(pos + (dash if draw else gap), length)
        if draw:
            xs, ys = int(x1 + dx * pos), int(y1 + dy * pos)
            xe, ye = int(x1 + dx * end), int(y1 + dy * end)
            cv2.line(frame, (xs, ys), (xe, ye), color, thickness)
        pos, draw = end, not draw


def _draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash=10, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    _draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash, gap)
    _draw_dashed_line(frame, (x1, y2), (x2, y2), color, thickness, dash, gap)
    _draw_dashed_line(frame, (x1, y1), (x1, y2), color, thickness, dash, gap)
    _draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash, gap)


def _save_comparison_video(preds_a, name_a, preds_b, name_b, source, exp_path, from_memory=True):
    COLOR_A = _COLOR_A_BGR   # azul sólido
    COLOR_B = _COLOR_B_BGR   # vermelho pontilhado

    def _load(i):
        if from_memory:
            return cv2.cvtColor(source[i].copy(), cv2.COLOR_RGB2BGR)
        return cv2.imread(source[i])

    frame0 = _load(0)
    if frame0 is None:
        return
    h, w = frame0.shape[:2]

    # escreve em /tmp (Linux fs) e move ao final — evita gargalo WSL2→NTFS
    tmp_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h),
    )

    for i in range(len(source)):
        frame = _load(i)
        if frame is None:
            continue
        if i < len(preds_a):
            x, y, bw, bh = map(int, preds_a[i])
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_A, 2)
        if i < len(preds_b):
            x, y, bw, bh = map(int, preds_b[i])
            _draw_dashed_rect(frame, (x, y), (x + bw, y + bh), COLOR_B, 2)
        _draw_legend(frame, _fmt_tracker(name_a), COLOR_A, _fmt_tracker(name_b), COLOR_B)
        writer.write(frame)

    writer.release()
    shutil.move(tmp_path, str(exp_path / "tracking.mp4"))


def _draw_legend(frame, name_a, color_a, name_b, color_b):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.5
    thickness = 1
    pad       = 6
    line_len  = 20

    (wa, ha), _ = cv2.getTextSize(name_a, font, scale, thickness)
    (wb, hb), _ = cv2.getTextSize(name_b, font, scale, thickness)
    box_w = max(wa, wb) + line_len + pad * 3
    box_h = ha + hb + pad * 3
    x0, y0 = 8, 8

    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 180),  1)

    y1 = y0 + pad + ha
    cv2.line(frame, (x0 + pad, y1 - ha // 2), (x0 + pad + line_len, y1 - ha // 2), color_a, 2)
    cv2.putText(frame, name_a, (x0 + pad + line_len + 4, y1), font, scale, (0, 0, 0), thickness)

    y2 = y1 + pad + hb
    _draw_dashed_line(frame, (x0 + pad, y2 - hb // 2), (x0 + pad + line_len, y2 - hb // 2), color_b, 2, dash=4, gap=3)
    cv2.putText(frame, name_b, (x0 + pad + line_len + 4, y2), font, scale, (0, 0, 0), thickness)


# ── experimento único ────────────────────────────────────────────────────────

def create_experiment_folder(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name.split("_")[1]) for p in root.glob("experimento_*") if p.is_dir()]
    next_id = (max(ids) + 1) if ids else 1
    path = root / f"experimento_{next_id}"
    path.mkdir()
    return path


def save_results(predictions, frames, errors, exec_time, exp_path, image_paths=None, save_video=True, save_plot=True):
    exp_path = Path(exp_path)
    metrics_path = exp_path / "metrics"
    metrics_path.mkdir(exist_ok=True)

    with open(exp_path / "bboxes.txt", "w") as f:
        for p in predictions:
            f.write(f"{p}\n")

    mean_err = np.mean(errors)
    max_err  = np.max(errors)
    fps      = len(predictions) / exec_time

    with open(metrics_path / "metrics.txt", "w") as f:
        f.write(f"Arquivo: {exp_path}\n")
        f.write(f"Erro medio: {mean_err}\n")
        f.write(f"Erro maximo: {max_err}\n")
        f.write(f"FPS: {fps}\n")

    with open(metrics_path / "errors.txt", "w") as f:
        for i, e in enumerate(errors):
            f.write(f"{frames[i]},{e}\n")

    if save_plot:
        fig, ax = plt.subplots()
        ax.plot(frames, errors)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Erro (px)")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(metrics_path / "error.png")
        plt.close(fig)

    if save_video and image_paths:
        frame0 = cv2.imread(image_paths[0])
        h, w = frame0.shape[:2]
        writer = cv2.VideoWriter(
            str(exp_path / "tracking.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h),
        )
        for i, path in enumerate(image_paths):
            frame = cv2.imread(path)
            if i < len(predictions):
                x, y, bw, bh = map(int, predictions[i])
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            writer.write(frame)
        writer.release()


def get_top_n_experiments(results_root, n=5):
    results = []
    for folder in sorted(Path(results_root).glob("experimento_*")):
        metrics_file = folder / "metrics" / "metrics.txt"
        params_file  = folder / "params.json"
        if not metrics_file.exists() or not params_file.exists():
            continue
        mean_error = None
        with open(metrics_file) as f:
            for line in f:
                if line.startswith("Erro medio:"):
                    mean_error = float(line.split(":")[1].strip())
                    break
        if mean_error is None:
            continue
        with open(params_file) as f:
            params = json.load(f)
        exp_id = int(folder.name.split("_")[1])
        results.append((exp_id, mean_error, params))
    results.sort(key=lambda x: x[1])
    return results[:n]
