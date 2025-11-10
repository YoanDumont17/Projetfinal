# -*- coding: utf-8 -*-
"""
Analyse du contraste vidéo (1 img/s)
- α : contraste latéral
- ω : contraste transversal
Détecte la première fois qu’un paramètre dépasse 0.2,
puis identifie le maximum de ce paramètre tant qu’il reste au-dessus du seuil.
S’arrête dès que le paramètre retombe sous le seuil.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== PARAMÈTRES ======
VIDEO_PATH = Path("/Users/vincentlelievre/Desktop/video1balayagehorsfocale.avi")  # <- ta vidéo
ALPHA_WEIGHT = 1.0
OMEGA_WEIGHT = 1.0
THRESHOLD = 0.2
MAX_SECONDS = None
SHOW_PARTS = True
DPI = 150
GAUSS_BLUR = True
SMOOTH_1D = True

def smooth1d(x):
    if not SMOOTH_1D or x.size < 5:
        return x
    k = np.array([1,4,6,4,1], dtype=np.float32) / 16.0
    return np.convolve(x, k, mode="same")

def compute_alpha(gray):
    col_means = gray.mean(axis=0).astype(np.float32)
    col_means = smooth1d(col_means)
    return float(col_means.std(ddof=0))

def compute_omega(gray):
    row_means = gray.mean(axis=1).astype(np.float32)
    row_means = smooth1d(row_means)
    return float(row_means.std(ddof=0))

def compute_score(gray):
    if GAUSS_BLUR:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    alpha = compute_alpha(gray)
    omega = compute_omega(gray)
    score = ALPHA_WEIGHT * alpha + OMEGA_WEIGHT * omega
    return score, alpha, omega

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Fichier vidéo introuvable: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if total_frames > 0 else 0.0
    if MAX_SECONDS is not None:
        duration_s = min(duration_s, MAX_SECONDS)

    times, alphas, omegas, frames = [], [], [], []
    n_seconds = int(np.floor(duration_s))

    # --- Variables de suivi ---
    detection_active = False
    param_name = None
    values = None
    idx_start = None
    best_value = -1.0
    best_time = 0.0
    best_frame = None

    for t in range(n_seconds + 1):
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        _, a, w = compute_score(gray)
        times.append(t)
        alphas.append(a)
        omegas.append(w)
        frames.append(frame.copy())

        # --- Étape 1 : déclenchement ---
        if not detection_active:
            if a > THRESHOLD or w > THRESHOLD:
                detection_active = True
                param_name = "α" if a > THRESHOLD else "ω"
                values = alphas if param_name == "α" else omegas
                idx_start = len(times) - 1
                best_value = values[idx_start]
                best_time = times[idx_start]
                best_frame = frames[idx_start]
                print(f"Détection du paramètre {param_name} à t = {best_time}s (valeur = {best_value:.3f})")
            continue

        # --- Étape 2 : zone active au-dessus du seuil ---
        current_val = values[-1]
        if current_val > THRESHOLD:
            # mise à jour du maximum dans la zone
            if current_val > best_value:
                best_value = current_val
                best_time = times[-1]
                best_frame = frames[-1]
        else:
            # le paramètre est retombé sous le seuil → on s'arrête ici
            idx_end = len(times) - 1
            print(f"→ Fin de la zone à t = {times[idx_end]}s")
            print(f"→ Valeur maximale de {param_name} = {best_value:.3f} à t = {best_time}s")
            break

    cap.release()

    if best_frame is None:
        print(f"Aucun dépassement du seuil {THRESHOLD} détecté.")
        return

    # --- Sauvegarde de la frame correspondante ---
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "images_test" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_frame = out_dir / f"{VIDEO_PATH.stem}_{param_name}_max_above_{THRESHOLD}_t{int(best_time)}s.png"
    cv2.imwrite(str(out_frame), best_frame)

    # --- Graphique ---
    fig = plt.figure(figsize=(9, 4.5))
    ax = plt.gca()
    ax.plot(times, alphas, linestyle="--", marker=".", label="α : contraste latéral")
    ax.plot(times, omegas, linestyle="dashdot", marker=".", label="ω : contraste transversal")
    ax.axhline(THRESHOLD, color="red", linestyle=":", alpha=0.6, label=f"Seuil {THRESHOLD}")
    ax.axvline(best_time, color="orange", linestyle="--", alpha=0.6,
            label=f"Max {param_name} à t={int(best_time)}s")

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Contraste normalisé")
    ax.set_title(f"Détection du premier dépassement ({param_name})")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()

    out_plot = out_dir / f"{VIDEO_PATH.stem}_threshold_{THRESHOLD}_{param_name}.png"
    plt.savefig(str(out_plot), dpi=DPI)
    plt.close(fig)

    print(f" Image sauvegardée : {out_frame}")

if __name__ == "__main__":
    main()