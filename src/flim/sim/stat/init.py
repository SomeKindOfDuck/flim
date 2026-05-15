"""
cli引数に基づいて各エージェントごとの設定ファイル（YAML）を作成する
または、すでに存在する場合は、モデルと訓練履歴を初期化する
"""

import argparse
import json
import os
import shutil
import sys
from os.path import exists


def parse_arg():
    parser = argparse.ArgumentParser(description="Set up configuration and results directory for given ID.")
    parser.add_argument("id", help="エージェントごとのID")
    parser.add_argument("--experiment", "-i", help="実験ID")
    parser.add_argument("--shift", type=int, nargs=2, help="視覚刺激を引数の範囲でランダムにずらす")
    parser.add_argument("--blur", type=float, nargs=2, help="視覚刺激に適用するgaussian blurの分散")
    parser.add_argument("--sequence", "-s", nargs="+", type=str, help="訓練に使用する刺激タイプとその順序")
    parser.add_argument("--epochs", "-e", nargs="+", type=int, help="sequenceの各要素ごとのエポック数")
    parser.add_argument("--sigma", nargs=2, type=float, help="モデルの各モダリティの内部ユニットのノイズ")

    args = parser.parse_args()
    return args

def init(args):
    run_id = args.id
    experiment_id = args.experiment

    default_config_path = "./config/default.json"
    if experiment_id is None:
        new_config_path = f"./config/{run_id}.json"
        results_dir = f"./results/{run_id}"
    else:
        new_config_dir = f"./config/{experiment_id}"
        os.makedirs(new_config_dir, exist_ok=True)
        new_config_path = os.path.join(new_config_dir, f"{run_id}.json")
        results_dir = f"./results/{experiment_id}/{run_id}"

    if not os.path.exists(default_config_path):
        print(f"Error: {default_config_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(default_config_path, "r") as f:
            config = json.load(f)
        config["id"] = run_id
    except Exception as e:
        print(f"Failed to process config JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if args.shift is not None:
        visual_shift, audio_shift = tuple(args.shift)
        config["noise"]["visual-shift"] = visual_shift
        config["noise"]["audio-shift"] = audio_shift
    if args.blur is not None:
        visual_blur, audio_blur = tuple(args.blur)
        config["noise"]["visual-blur"] = visual_blur
        config["noise"]["audio-blur"] = audio_blur

    if experiment_id is not None:
        config["experiment"] = experiment_id
    if args.sequence is not None:
        config["training"]["sequence"] = args.sequence
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.sigma is not None:
        sigma_v, sigma_a = args.sigma
        config["model"]["sigma_v"] = sigma_v
        config["model"]["sigma_a"] = sigma_a

    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Created config with updated ID: {new_config_path}")

    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    except Exception as e:
        print(f"Failed to create results directory: {e}", file=sys.stderr)
        sys.exit(1)


def reset(args):
    run_id = args.id
    experiment_id = args.experiment
    if experiment_id is not None:
        config_path = f"./config/{experiment_id}/{run_id}.json"
        results_dir = f"./results/{experiment_id}/{run_id}"
    else:
        config_path = f"./config/{run_id}.json"
        results_dir = f"./results/{run_id}"

    if os.path.exists(results_dir):
        print(f"Clearing contents of {results_dir}...")
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"{results_dir} does not exist. Creating it...")
        os.makedirs(results_dir)

    print(f"Agent '{run_id}' has been successfully reset.")

def main():
    args = parse_arg()

    run_id = args.id
    experiment_id = args.experiment
    if experiment_id is not None:
        config_path = f"./config/{experiment_id}/{run_id}.json"
    else:
        config_path = f"./config/{run_id}.json"

    if exists(config_path):
        reset(args)
    else:
        init(args)

if __name__ == "__main__":
    main()
