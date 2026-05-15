import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(vrecs, arecs, save_dir, panel_h=3, panel_w=6):

    os.makedirs(save_dir, exist_ok=True)

    grouped = defaultdict(list)
    for freq in vrecs.keys():
        v, a = freq
        if v > 0 and a > 0 and v == a:
            grouped["synchronous"].append(freq)
        elif v > 0 and a > 0 and v != a:
            grouped["asynchronous"].append(freq)
        elif v > 0 and a == 0:
            grouped["visual_only"].append(freq)
        elif v == 0 and a > 0:
            grouped["audio_only"].append(freq)

    for category, freqs in grouped.items():
        freqs = sorted(freqs)
        n_rows = len(freqs)
        fig, axes = plt.subplots(n_rows, 2, figsize=(2 * panel_w, panel_h * n_rows), sharex='col')

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, freq in enumerate(freqs):
            vis = vrecs[freq].numpy().T
            aud = arecs[freq].numpy().T

            ax_vis = axes[i, 0]
            ax_aud = axes[i, 1]

            im_vis = ax_vis.imshow(vis, aspect='auto', cmap='viridis', origin='lower')
            im_aud = ax_aud.imshow(aud, aspect='auto', cmap='viridis', origin='lower')

            ax_vis.set_title("Visual")
            ax_aud.set_title("Audio")

            if i == n_rows - 1:
                ax_vis.set_xlabel("Time step")
                ax_aud.set_xlabel("Time step")
            ax_vis.set_ylabel(f"{freq} Hz\nFeature idx")

            fig.colorbar(im_vis, ax=ax_vis, fraction=0.046, pad=0.04)
            fig.colorbar(im_aud, ax=ax_aud, fraction=0.046, pad=0.04)

        plt.tight_layout()
        savepath = os.path.join(save_dir, f"reconstruction_{category}.png")
        plt.savefig(savepath)
        savepath = os.path.join(save_dir, f"reconstruction_{category}.pdf")
        plt.savefig(savepath)
        plt.close()

        print(f"Saved VAE reconstruction figures in {savepath}")


def plot_reward_trajectories(mean_reward_preds, result_dir):
    from collections import defaultdict

    # 刺激分類
    categories = {
        "synchronous": [],
        "asynchronous": [],
        "visual_only": [],
        "audio_only": [],
    }
    for f in mean_reward_preds:
        v, a = f
        if v > 0 and a > 0 and v == a:
            categories["synchronous"].append(f)
        elif v > 0 and a > 0 and v != a:
            categories["asynchronous"].append(f)
        elif v > 0 and a == 0:
            categories["visual_only"].append(f)
        elif v == 0 and a > 0:
            categories["audio_only"].append(f)

    cmap = plt.get_cmap("viridis")

    for cat, freqs in categories.items():
        if not freqs:
            continue
        freqs = sorted(freqs)

        color_key_idx = 0 if cat in ["synchronous", "visual_only"] else 1
        color_vals = [f[color_key_idx] for f in freqs]
        norm = plt.Normalize(vmin=min(color_vals), vmax=max(color_vals))

        plt.figure(figsize=(8, 5))
        for f in freqs:
            y = mean_reward_preds[f].numpy()
            color = cmap(norm(f[color_key_idx]))
            plt.plot(y, color=color, label=f"{f} Hz")
            plt.axhline(y[-100:].mean(), color=color, linestyle='--', linewidth=1.5)
        plt.axhline(0.5, color='black', linestyle='--', linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel("Predicted reward probability")
        plt.title(f"Reward Trajectory — {cat.replace('_', ' ').capitalize()} stimuli")
        plt.legend(loc='lower left')
        plt.tight_layout()
        savepath = f"{result_dir}/reward_trajectory_{cat}.png"
        plt.savefig(savepath)
        savepath = f"{result_dir}/reward_trajectory_{cat}.pdf"
        plt.savefig(savepath)
        print(f"Saved reward trajectory of {cat} flickr in {savepath}")
        plt.close()


def plot_psychometric_function(mean_reward_preds, result_dir, show):
    import pandas as pd

    # 分類と設定
    marker_settings = {
        "synchronous": {
            "cond": lambda f: f[0] > 0 and f[1] > 0 and f[0] == f[1],
            "x": lambda f: f[0],
            "color_val": lambda f: f[0],
            "marker": "o"
        },
        "visual_only": {
            "cond": lambda f: f[0] > 0 and f[1] == 0,
            "x": lambda f: f[0],
            "color_val": lambda f: f[0],
            "marker": "s"
        },
        "audio_only": {
            "cond": lambda f: f[0] == 0 and f[1] > 0,
            "x": lambda f: f[1],
            "color_val": lambda f: f[1],
            "marker": "^"
        },
        "asynchronous": {
            "cond": lambda f: f[0] > 0 and f[1] > 0 and f[0] != f[1],
            "x": lambda f: f[1],
            "color_val": lambda f: f[1],
            "marker": "D"
        }
    }

    # カラーマップとプロット
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8, 6))

    all_color_vals = [
        settings["color_val"](f)
        for label, settings in marker_settings.items()
        for f in mean_reward_preds if settings["cond"](f)
    ]
    if not all_color_vals:
        print("No data to plot.")
        return

    norm = plt.Normalize(vmin=min(all_color_vals), vmax=max(all_color_vals))

    data_rows = []

    for label, settings in marker_settings.items():
        freqs = [f for f in mean_reward_preds if settings["cond"](f)]
        if not freqs:
            continue
        x_vals = [settings["x"](f) for f in freqs]
        y_vals = [mean_reward_preds[f].mean().item() for f in freqs]
        colors = [cmap(norm(settings["color_val"](f))) for f in freqs]

        ax.scatter(
            x_vals, y_vals,
            c=colors,
            s=80,
            edgecolor='k',
            marker=settings["marker"],
            label=label.replace('_', ' ').capitalize()
        )

        for f, y in zip(freqs, y_vals):
            data_rows.append({
                "stimulus_type": label,
                "visual_freq": f[0],
                "audio_freq": f[1],
                "reward_pred": y
            })

    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(result_dir, "reward_preds.csv")
    df.to_csv(csv_path, index=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Flicker Frequency (Hz)")
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Flicker Frequency (Hz)")
    ax.set_ylabel("Mean Predicted Reward")
    ax.set_title("Mean Reward Prediction by Frequency and Stimulus Type")
    ax.grid(alpha=0.3)
    ax.set_ylim((0.0, 1.0))
    ax.legend()
    plt.tight_layout()
    savepath = os.path.join(result_dir, f"reward_vs_frequency.png")
    plt.savefig(savepath)
    savepath = os.path.join(result_dir, f"reward_vs_frequency.pdf")
    plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

    print(f"Saved psychometric function plot in {savepath}")
    print(f"Saved reward predictions CSV in {csv_path}")


def plot_dynamic_perception_grouped(results, save_dir, panel_w=6, panel_h=3):
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    from flim.stimgen import _N

    I = _N ** 2

    os.makedirs(save_dir, exist_ok=True)

    grouped = {
        "synchronous": [],
        "asynchronous": [],
        "visual_only": [],
        "audio_only": []
    }

    for key in results:
        v, a = key
        if v > 0 and a > 0 and v == a:
            grouped["synchronous"].append(key)
        elif v > 0 and a > 0 and v != a:
            grouped["asynchronous"].append(key)
        elif v > 0 and a == 0:
            grouped["visual_only"].append(key)
        elif v == 0 and a > 0:
            grouped["audio_only"].append(key)

    # 各カテゴリごとにプロット
    for category, keys in grouped.items():
        if not keys:
            continue

        n_rows = len(keys)
        fig, axes = plt.subplots(n_rows, 4, figsize=(2 * panel_w, n_rows * panel_h), sharex=True)

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, key in enumerate(sorted(keys)):
            res = results[key]

            vis_blend = res["v_blend"].reshape(-1, I)
            aud_blend = res["a_blend"].reshape(-1, I)
            vis_pred  = res["v_pred"].reshape(-1, I)
            aud_pred  = res["a_pred"].reshape(-1, I)

            axes[row, 0].imshow(vis_pred,  aspect='auto', cmap='viridis', origin='lower')
            axes[row, 1].imshow(vis_blend, aspect='auto', cmap='viridis', origin='lower')
            axes[row, 2].imshow(aud_pred,  aspect='auto', cmap='viridis', origin='lower')
            axes[row, 3].imshow(aud_blend, aspect='auto', cmap='viridis', origin='lower')

            freq_label = f"v={key[0]}, a={key[1]}"
            axes[row, 0].set_ylabel(freq_label)

            if row == 0:
                axes[0, 0].set_title("Visual prediction")
                axes[0, 1].set_title("Visual integration")
                axes[0, 2].set_title("Audio prediction")
                axes[0, 3].set_title("Audio integration")

        plt.tight_layout()
        savepath = os.path.join(save_dir, f"dynamic_perception_{category}.png")
        plt.savefig(savepath)
        savepath = os.path.join(save_dir, f"dynamic_perception_{category}.pdf")
        plt.savefig(savepath)
        plt.close()
        print(f"Saved: {savepath}")
