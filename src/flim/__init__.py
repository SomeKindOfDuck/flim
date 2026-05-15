from torch import nn
from torch.utils.data import DataLoader


def fmt_2dig(x):
    return f"{x: 0.2f}".lstrip()


def name_savedir(result_dir: str, train_method: str):
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join(result_dir, f"{train_method}/{timestamp}")


def append_training_log(result_dir, file, train_config):
    import os
    from datetime import datetime

    log_path = os.path.join(result_dir, "log.txt")
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    epoch = train_config.get("epoch", "N/A")
    lr = train_config.get("lr", "N/A")
    log_entry = f"{timestamp} : {os.path.basename(file)} : epoch={epoch} : lr={lr}\n"

    # 書き込み（追記モード）
    with open(log_path, "a") as f:
        f.write(log_entry)

    print(f"Appended log to {log_path}")
