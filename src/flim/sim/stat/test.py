"""
このプラグラムはモデルの非同期刺激によるテストを実行する
訓練経過とテストじのデコーダによる視聴覚入力の復元と報酬予測のpsychometric functionの図を作成する
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import flim.sim.stat.logplot as plt
from flim import append_training_log, name_savedir
from flim.model.bhv import GRURewardPredictor
from flim.sim.stat import average_across_loader, spawn_model, train
from flim.stimgen import (_N, FLKLDataset, gaussian_smoothing,
                          generate_asynchronous_dataset,
                          generate_audio_dataset, generate_synchronous_dataset,
                          generate_visual_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=str, help="エージェントID")
    parser.add_argument("--plot", "-p", action="store_true", help="訓練の途中経過の図を保存する")
    args = parser.parse_args()

    for config in args.configs:
        print(f"Start training with {config}.")
        config_path = config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        run_id = config.get("id")
        experiment_id = config.get("experiment")

        cmvae, reward_predictor = spawn_model(run_id, config)
        lr = config["model"]["lr"]

        if experiment_id is not None:
            result_dir = f"./results/{experiment_id}/{run_id}"
        else:
            result_dir = f"./results/{run_id}"
        os.makedirs(result_dir, exist_ok=True)

        noise_config = config["noise"]

        visual_shift = noise_config.get("visual-shift")
        audio_shift = noise_config.get("audio-shift")
        visual_blur = noise_config.get("visual-blur")
        audio_blur = noise_config.get("audio-blur")

        test_config = config["test"]

        train_datalist = []
        test_datalist = []

        mess = "all types of"
        condition = "Async"

        # データ生成と分割
        train_datalist += generate_asynchronous_dataset(test_config["asynchronous-flickr"], rep=20, vsft=visual_shift, asft=audio_shift)
        train_datalist += generate_visual_dataset(test_config["visual-flickr"], rep=20, vsft=visual_shift)
        train_datalist += generate_audio_dataset(test_config["audio-flickr"], rep=20, asft=audio_shift)
        train_datalist += generate_synchronous_dataset(test_config["synchronous-flickr"], rep=20, vsft=visual_shift, asft=audio_shift)

        train_datalist = gaussian_smoothing(train_datalist, visual_blur, audio_blur)

        train_loader = DataLoader(FLKLDataset(train_datalist), batch_size=1, shuffle=True, num_workers=0)

        print("Running only evaluation.")
        append_training_log(result_dir, condition, { "epoch": None, "lr": None })

        mean_visual_recs, mean_audio_recs, mean_reward_preds = average_across_loader(cmvae, reward_predictor, train_loader)
        savefig_dir = name_savedir(result_dir, condition)
        plt.plot_reconstruction(mean_visual_recs, mean_audio_recs, savefig_dir)
        plt.plot_reward_trajectories(mean_reward_preds, savefig_dir)
        plt.plot_psychometric_function(mean_reward_preds, savefig_dir, args.plot)

        print(f"Evaluation complete. Results saved to {result_dir}")

if __name__ == "__main__":
    main()
