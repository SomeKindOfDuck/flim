"""
このプラグラムはモデルの訓練と非同期刺激によるテストを実行する
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
    parser.add_argument("--modality", "-m", nargs="+", type=str, help="訓練に使用するモダリティ(synchronous, visual, audio)")
    parser.add_argument("--epoch", "-e", type=int, help="訓練のエポック数")
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

        train_config = config["training"]

        train_seq = train_config.get("sequence")
        epochs = train_config.get("epochs")

        noise_config = config["noise"]

        visual_shift = noise_config.get("visual-shift")
        audio_shift = noise_config.get("audio-shift")
        visual_blur = noise_config.get("visual-blur")
        audio_blur = noise_config.get("audio-blur")

        print(f"Shifting the flicker pulses (visual={visual_shift}; auditory={audio_shift})")
        print(args.modality)

        train_datalist = []
        test_datalist = []
        mess = ""
        condition = ""

        # データ生成と分割
        if "visual" in args.modality:
            flickr_freqs = train_config["visual-flickr"]
            train_datalist += generate_visual_dataset(flickr_freqs, rep=20, vsft=visual_shift)
            test_datalist += generate_visual_dataset(flickr_freqs, rep=5, vsft=visual_shift)
            mess += "visual, "
            condition += "Vis"

        if "audio" in args.modality:
            flickr_freqs = train_config["audio-flickr"]
            train_datalist += generate_audio_dataset(flickr_freqs, rep=20, asft=audio_shift)
            test_datalist += generate_audio_dataset(flickr_freqs, rep=5, asft=audio_shift)
            mess += "audio, "
            condition += "Aud"

        if "synchronous" in args.modality:
            flickr_freqs = train_config["synchronous-flickr"]
            train_datalist += generate_synchronous_dataset(flickr_freqs, rep=20, vsft=visual_shift, asft=audio_shift)
            test_datalist += generate_synchronous_dataset(flickr_freqs, rep=5, vsft=visual_shift, asft=audio_shift)
            mess += "synchronous, "
            condition += "Sync"

        train_datalist = gaussian_smoothing(train_datalist, visual_blur, audio_blur)
        test_datalist = gaussian_smoothing(test_datalist, visual_blur, audio_blur)

        train_loader = DataLoader(FLKLDataset(train_datalist), batch_size=1, shuffle=True, num_workers=0)
        test_loader = DataLoader(FLKLDataset(test_datalist), batch_size=1, shuffle=True, num_workers=0)

        # 学習
        epoch = args.epoch
        log_content = {}
        print(f"Start training with {mess[0:-2]} flickrs for {epoch} epochs.")
        train_log = train(cmvae, reward_predictor, train_loader, test_loader, epoch, lr, config["model"].get("kl-beta", 1.0))
        log_content["epoch"] = epoch
        log_content["lr"] = lr
        print(f"Training complete. Results saved to {result_dir}")
        # モデル保存
        torch.save(cmvae.state_dict(), f"{result_dir}/cmvae.pth")
        torch.save(reward_predictor.state_dict(), f"{result_dir}/reward_predictor.pth")
        append_training_log(result_dir, condition, log_content)

        # 結果の描画 & 保存
        mean_visual_recs, mean_audio_recs, mean_reward_preds = average_across_loader(cmvae, reward_predictor, train_loader)
        savefig_dir = name_savedir(result_dir, condition)
        plt.plot_reconstruction(mean_visual_recs, mean_audio_recs, savefig_dir)
        plt.plot_reward_trajectories(mean_reward_preds, savefig_dir)
        plt.plot_psychometric_function(mean_reward_preds, savefig_dir, args.plot)

        print(f"Evaluation complete. Results saved to {result_dir}")

if __name__ == "__main__":
    main()
