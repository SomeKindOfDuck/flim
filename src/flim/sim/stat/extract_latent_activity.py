# -*- coding: utf-8 -*-

"""
訓練済みのモデルに入力を与えたときのユニット活動や埋め込みベクトルを
変数ごとに別ファイルのWIDE形式で出力する
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flim.sim.stat import spawn_model
from flim.stimgen import (_N, FLKLDataset, gaussian_smoothing,
                          generate_asynchronous_dataset,
                          generate_audio_dataset, generate_synchronous_dataset,
                          generate_visual_dataset)

I = _N ** 2
DEVICE = torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract all-trial activities and save each variable as a separate WIDE-format file."
    )
    p.add_argument("config", help="Path to config json (single file).")
    return p.parse_args()


def build_dataset_from_config(config: Dict[str, Any]):
    test_config = config["test"]
    noise = config.get("noise", {})

    # vsft = noise.get("visual-shift")
    # asft = noise.get("audio-shift")
    vsft = 0
    asft = 0
    vblur = noise.get("visual-blur", 0)
    ablur = noise.get("audio-blur", 0)
    # vblur = 0
    # ablur = 0

    rep = 5

    datalist = []
    datalist += generate_asynchronous_dataset([1, 4, 14, 17], rep=rep, vsft=vsft, asft=asft)
    datalist += generate_visual_dataset([4, 6, 9, 12, 14], rep=rep, vsft=vsft)
    datalist += generate_audio_dataset([4, 9, 14], rep=rep, asft=asft)
    datalist += generate_synchronous_dataset([4, 6, 9, 12, 14], rep=rep, vsft=vsft, asft=asft)

    if vblur > 0 or ablur > 0:
        dataset = gaussian_smoothing(datalist, vblur, ablur)

    return FLKLDataset(datalist)


@torch.no_grad()
def extract_trial_arrays(
    cmvae, reward_predictor, vobs: torch.Tensor, aobs: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    vobs, aobs: (T, I) float32 tensors on CPU
    Returns:
      - collapsed inputs: vobs_sum, aobs_sum (T,)
      - VAE unit activities (h only): enc_v_h, enc_a_h, enc_cross_h, dec_*_h* (T, D)
      - latent: z (T, Z)
      - collapsed recon: v_recon_sum, a_recon_sum (T,)
      - RNN: gru_h (T, H), reward_logit, reward_pred (T,)
    """
    T = vobs.shape[0]
    enc = cmvae.encoder
    dec = cmvae.decoder

    enc_v_h, enc_a_h, enc_cross_h, z_list = [], [], [], []
    dec_v_h1, dec_v_h2, dec_a_h1, dec_a_h2 = [], [], [], []

    v_recon_sum = np.zeros(T, dtype=np.float32)
    a_recon_sum = np.zeros(T, dtype=np.float32)

    for t in range(T):
        vt = vobs[t : t + 1]
        at = aobs[t : t + 1]

        vh = F.relu(enc.visual_in(vt))
        ah = F.relu(enc.audio_in(at))
        ch = F.relu(enc.hidden(torch.cat((vh, ah), dim=1)))

        mu = enc.mu(ch)
        sigma = F.softplus(enc.sigma(ch))
        eps = torch.randn_like(mu)
        z = mu + sigma.div(2).exp() * eps

        dvh1 = F.relu(dec.visual_hidden1(z))
        dah1 = F.relu(dec.audio_hidden1(z))
        dvh2 = F.relu(dec.visual_hidden2(dvh1))
        dah2 = F.relu(dec.audio_hidden2(dah1))

        vrec = torch.sigmoid(dec.visual_out(dvh2))
        arec = torch.sigmoid(dec.audio_out(dah2))

        enc_v_h.append(vh.squeeze(0).numpy())
        enc_a_h.append(ah.squeeze(0).numpy())
        enc_cross_h.append(ch.squeeze(0).numpy())
        z_list.append(z.squeeze(0).numpy())

        dec_v_h1.append(dvh1.squeeze(0).numpy())
        dec_v_h2.append(dvh2.squeeze(0).numpy())
        dec_a_h1.append(dah1.squeeze(0).numpy())
        dec_a_h2.append(dah2.squeeze(0).numpy())

        v_recon_sum[t] = float(vrec.sum().item())
        a_recon_sum[t] = float(arec.sum().item())

    z_arr = np.stack(z_list, axis=0).astype(np.float32)
    z_seq = torch.from_numpy(z_arr).unsqueeze(0)

    rnn_out, _ = reward_predictor.rnn(z_seq)
    logits = reward_predictor.fc(rnn_out).squeeze(-1)
    pred = torch.sigmoid(logits)

    return {
        "vobs_sum": vobs.sum(dim=1).numpy().astype(np.float16),
        "aobs_sum": aobs.sum(dim=1).numpy().astype(np.float16),
        "enc_v_h": np.stack(enc_v_h, axis=0).astype(np.float32),
        "enc_a_h": np.stack(enc_a_h, axis=0).astype(np.float32),
        "enc_cross_h": np.stack(enc_cross_h, axis=0).astype(np.float32),
        "z": z_arr,
        "dec_v_h1": np.stack(dec_v_h1, axis=0).astype(np.float32),
        "dec_v_h2": np.stack(dec_v_h2, axis=0).astype(np.float32),
        "dec_a_h1": np.stack(dec_a_h1, axis=0).astype(np.float32),
        "dec_a_h2": np.stack(dec_a_h2, axis=0).astype(np.float32),
        "v_recon_sum": v_recon_sum,
        "a_recon_sum": a_recon_sum,
        "gru_h": rnn_out.squeeze(0).detach().numpy().astype(np.float32),
        "reward_logit": logits.squeeze(0).detach().numpy().astype(np.float16),
        "reward_pred": pred.squeeze(0).detach().numpy().astype(np.float16),
    }


def base_signals_df(
    arrays: Dict[str, np.ndarray],
    robs: np.ndarray,
    trial_idx: int,
    v_freq: float,
    a_freq: float,
) -> pd.DataFrame:
    T = int(arrays["reward_pred"].shape[0])
    df = pd.DataFrame(
        {
            "trial_idx": trial_idx,
            "time": np.arange(T, dtype=np.int32),
            "v_freq": v_freq,
            "a_freq": a_freq,
            "vobs_sum": arrays["vobs_sum"],
            "aobs_sum": arrays["aobs_sum"],
            # "v_recon_sum": arrays["v_recon_sum"],
            # "a_recon_sum": arrays["a_recon_sum"],
            # "reward_logit": arrays["reward_logit"],
            # "reward_pred": arrays["reward_pred"],
            # "robs": robs.astype(np.float32),
        }
    )
    return df

def activity_wide_df(
    mat: np.ndarray,
    trial_idx: int,
    v_freq: float,
    a_freq: float,
    module: str,
    submodule: str,
    name: str,
    vobs_sum: np.ndarray,
    aobs_sum: np.ndarray,
) -> pd.DataFrame:
    if mat.ndim == 1:
        mat = mat[:, None]
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {mat.shape}")

    T, D = mat.shape
    unit_cols = [f"unit_{i:03d}" for i in range(D)]

    df = pd.DataFrame(mat, columns=unit_cols)

    df.insert(0, "aobs_sum", aobs_sum)
    df.insert(0, "vobs_sum", vobs_sum)
    df.insert(0, "name", name)
    df.insert(0, "submodule", submodule)
    df.insert(0, "module", module)
    df.insert(0, "a_freq", a_freq)
    df.insert(0, "v_freq", v_freq)
    df.insert(0, "time", np.arange(T, dtype=np.int32))
    df.insert(0, "trial_idx", trial_idx)

    return df


def variable_specs():
    return {
        "enc_v_h": {
            "module": "CMVAE",
            "submodule": "visual-encoder",
            "name": "enc_v_h",
        },
        "enc_a_h": {
            "module": "CMVAE",
            "submodule": "audio-encoder",
            "name": "enc_a_h",
        },
        "enc_cross_h": {
            "module": "CMVAE",
            "submodule": "cross-encoder",
            "name": "enc_cross_h",
        },
        "z": {
            "module": "CMVAE",
            "submodule": "latent",
            "name": "z",
        },
        "dec_v_h1": {
            "module": "CMVAE",
            "submodule": "visual-decoder-1",
            "name": "dec_v_h1",
        },

        "dec_v_h2": {
            "module": "CMVAE",
            "submodule": "visual-decoder-2",
            "name": "dec_v_h2",
        },
        "dec_a_h1": {
            "module": "CMVAE",
            "submodule": "audio-decoder-1",
            "name": "dec_a_h1",
        },
        "dec_a_h2": {
            "module": "CMVAE",
            "submodule": "audio-decoder-2",
            "name": "dec_a_h2",
        },
        "gru_h": {
            "module": "RNN",
            "submodule": "gru",
            "name": "gru_h",
        },
        "reward_logit_only": {
            "module": "RNN",
            "submodule": "readout-logit",
            "name": "reward_logit",
        },
        "reward_pred_only": {
            "module": "RNN",
            "submodule": "readout-pred",
            "name": "reward_pred",
        },
    }


def save_dataframe_pair(df: pd.DataFrame, out_dir: str, stem: str):
    parquet_path = os.path.join(out_dir, f"{stem}.parquet")
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[OK] saved: {parquet_path}")
    print(f"[OK] saved: {csv_path}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    run_id = config["id"]
    experiment_id = config.get("experiment")

    cmvae, grup = spawn_model(run_id, config)
    cmvae.to(DEVICE).eval()
    grup.to(DEVICE).eval()

    dataset = build_dataset_from_config(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    specs = variable_specs()

    all_base_rows: List[pd.DataFrame] = []
    all_var_rows: Dict[str, List[pd.DataFrame]] = {
        "enc_v_h": [],
        "enc_a_h": [],
        "enc_cross_h": [],
        "z": [],
        "dec_v_h1": [],
        "dec_v_h2": [],
        "dec_a_h1": [],
        "dec_a_h2": [],
        "gru_h": [],
        "reward_logit": [],
        "reward_pred": [],
    }

    for idx, ((vobs, vp), (aobs, ap), robs, v_freq, a_freq) in enumerate(loader):
        v_freq = float(v_freq.item())
        a_freq = float(a_freq.item())

        vobs = vobs.view(-1, I).float()
        aobs = aobs.view(-1, I).float()
        robs_np = robs.view(-1).numpy()

        arrays = extract_trial_arrays(cmvae, grup, vobs, aobs)

        vobs_sum = vp.view(-1).float()
        aobs_sum = ap.view(-1).float()

        all_base_rows.append(
            base_signals_df(
                arrays=arrays,
                robs=robs_np,
                trial_idx=idx,
                v_freq=v_freq,
                a_freq=a_freq,
            )
        )

        for key in [
            "enc_v_h",
            "enc_a_h",
            "enc_cross_h",
            "z",
            "dec_v_h1",
            "dec_v_h2",
            "dec_a_h1",
            "dec_a_h2",
            "gru_h",
        ]:
            meta = specs[key]
            all_var_rows[key].append(
                activity_wide_df(
                    mat=arrays[key],
                    trial_idx=idx,
                    v_freq=v_freq,
                    a_freq=a_freq,
                    module=meta["module"],
                    submodule=meta["submodule"],
                    name=meta["name"],
                    vobs_sum=vobs_sum,
                    aobs_sum=aobs_sum,
                )
            )

        meta = specs["reward_logit_only"]
        all_var_rows["reward_logit"].append(
            activity_wide_df(
                mat=arrays["reward_logit"],
                trial_idx=idx,
                v_freq=v_freq,
                a_freq=a_freq,
                module=meta["module"],
                submodule=meta["submodule"],
                name=meta["name"],
                vobs_sum=vobs_sum,
                aobs_sum=aobs_sum,
            )
        )

        meta = specs["reward_pred_only"]
        all_var_rows["reward_pred"].append(
            activity_wide_df(
                mat=arrays["reward_pred"],
                trial_idx=idx,
                v_freq=v_freq,
                a_freq=a_freq,
                module=meta["module"],
                submodule=meta["submodule"],
                name=meta["name"],
                vobs_sum=vobs_sum,
                aobs_sum=aobs_sum,
            )
        )

    if experiment_id is not None:
        out_dir = os.path.join(".", "results", experiment_id, run_id, "activity")
    else:
        out_dir = os.path.join(".", "results", run_id, "activity")
    os.makedirs(out_dir, exist_ok=True)

    df_base = pd.concat(all_base_rows, axis=0, ignore_index=True)
    save_dataframe_pair(df_base, out_dir, f"base_signals")

    for key, dfs in all_var_rows.items():
        df = pd.concat(dfs, axis=0, ignore_index=True)
        save_dataframe_pair(df, out_dir, key)

    print(f"[OK] Experiment    : {experiment_id}")
    print(f"[OK] Agent id      : {run_id}")


if __name__ == "__main__":
    main()
