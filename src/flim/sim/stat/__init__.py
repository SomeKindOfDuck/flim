"""
モデルの初期化や訓練、訓練後のモデルの挙動を確認するための関数を提供する
"""
from torch import nn
from torch.utils.data import DataLoader

from flim import fmt_2dig


def spawn_model(id: str, config: dict):
    """
    dictで読み込んだ設定ファイル（config.yaml）から、訓練済みのモデルがあればロードし、なければ初期化する
    """
    import os

    import torch

    from flim.model.bhv import GRURewardPredictor
    from flim.model.prcpt import CrossModalVAE
    from flim.stimgen import _N

    I = _N ** 2


    mconf = config["model"]
    vae_h1 = mconf["CMVAE-Layer01"]
    vae_h2 = mconf["CMVAE-Layer02"]
    vae_z = mconf["CMVAE-Latent"]
    sigma_v = mconf.get("sigma_v", 0.)
    sigma_a = mconf.get("sigma_a", 0.)
    rew_rnn_h = mconf["RewardRNN"]

    cmvae = CrossModalVAE(I, vae_h1, vae_h2, vae_z, sigma_v, sigma_a)
    reward_predictor = GRURewardPredictor(vae_z, rew_rnn_h)

    experiment_id = config.get("experiment")
    if experiment_id is not None:
        result_dir = f"./results/{experiment_id}/{id}"
    else:
        result_dir = f"./results/{id}"
    cvae_path = os.path.join(result_dir, "cmvae.pth")
    gru_path = os.path.join(result_dir, "reward_predictor.pth")

    if os.path.exists(cvae_path) and os.path.exists(gru_path):
        cmvae.load_state_dict(torch.load(cvae_path, map_location='cpu', weights_only=True))
        reward_predictor.load_state_dict(torch.load(gru_path, map_location='cpu', weights_only=True))
        print(f"Loaded pretrained models from {result_dir}")
    else:
        print(f"No pretrained models found in {result_dir}. Initializing new models.")
    return cmvae, reward_predictor


def train(cmvae, reward_predictor, train_loader, test_loader, epoch: int, lr: float):
    import torch
    from torch.utils.data import DataLoader, random_split

    from flim.model.bhv import reward_loss
    from flim.model.prcpt import cmvae_reconstruction_loss, kl_to_prior_loss

    optimizer = torch.optim.Adam(list(cmvae.parameters()) + list(reward_predictor.parameters()), lr=lr)

    _, w, h = train_loader.dataset[0][0][0].shape
    I = w * h

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    history = {
        "train-loss": [],
        "val-loss": []
    }

    for epoch in range(epoch):
        cmvae.train()
        reward_predictor.train()

        loss_total, loss_vis, loss_aud, loss_kl, loss_rpe = None, None, None, None, None
        for i, ((vobs, _), (aobs, _), robs, _, _) in enumerate(train_loader):
            vobs = vobs.to(device).view(-1, I).to(torch.float32)
            aobs = aobs.to(device).view(-1, I).to(torch.float32)
            robs = robs.to(device).view(-1).to(torch.float32)

            visual_out, audio_out, mu, sigma, z = cmvae(vobs, aobs)
            reward_out = reward_predictor(z)

            loss_vis, loss_aud = cmvae_reconstruction_loss(vobs, aobs, visual_out, audio_out)
            loss_kl = kl_to_prior_loss(mu, sigma)
            loss_rpe = reward_loss(reward_out, robs)
            loss_total = loss_vis + loss_aud + loss_kl + loss_rpe

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            history["train-loss"].append(float(loss_total))

        cmvae.eval()
        reward_predictor.eval()
        with torch.no_grad():
            for i, ((vobs, _), (aobs, _), robs, _, _) in enumerate(test_loader):
                vobs = vobs.to(device).view(-1, I).to(torch.float32)
                aobs = aobs.to(device).view(-1, I).to(torch.float32)
                robs = robs.to(device).view(-1).to(torch.float32)

                visual_out, audio_out, mu, sigma, z = cmvae(vobs, aobs)
                reward_out = reward_predictor(z)

                loss_vis, loss_aud = cmvae_reconstruction_loss(vobs, aobs, visual_out, audio_out)
                loss_kl = kl_to_prior_loss(mu, sigma)
                loss_rpe = reward_loss(reward_out, robs)
                loss_total = loss_vis + loss_aud + loss_kl + loss_rpe

                history["val-loss"].append(float(loss_total))

            epoch_display = f"{epoch+1} epoch: "
            loss_display = f"total-loss: {fmt_2dig(loss_total)} (V: {fmt_2dig(loss_vis)}; A: {fmt_2dig(loss_aud)}; KL: {fmt_2dig(loss_kl)}; R: {fmt_2dig(loss_rpe)}"
            print(epoch_display + loss_display)
    return history


def average_across_loader(pmodel: nn.Module, bmodel: nn.Module, data_loader: DataLoader):
    """
    訓練済みのモデルにデータを入力した際の、視聴覚入力のデコードと報酬予測を生成する
    """
    from collections import defaultdict

    import torch

    from flim.stimgen import _N

    _, w, h = data_loader.dataset[0][0][0].shape
    I = w * h

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    pmodel.eval()
    bmodel.eval()

    v_preds, a_preds, r_preds = defaultdict(list), defaultdict(list), defaultdict(list)

    for (vobs, _), (aobs, _), _, vl, al in data_loader:
        vobs = vobs.to(device).view(-1, I).to(torch.float32)
        aobs = aobs.to(device).view(-1, I).to(torch.float32)

        if isinstance(vl, torch.Tensor):
            val = int(vl.item()), int(al.item())
        else:
            val = int(vl), int(al.item())

        vobs = vobs.squeeze(0) if vobs.dim() == 3 else vobs
        aobs = aobs.squeeze(0) if aobs.dim() == 3 else aobs
        T, Fv = vobs.shape

        vobs = vobs.to(device).float().view(-1, Fv)
        aobs = aobs.to(device).float().view(-1, Fv)

        with torch.no_grad():
            v_pred, a_pred, mu, sigma, z = pmodel(vobs, aobs)
            z = z.view(1, T, -1)

            r_pred = bmodel(z)
            r_pred = r_pred.squeeze(0).cpu()

        v_preds[val].append(v_pred.cpu().view(T, Fv))
        a_preds[val].append(a_pred.cpu().view(T, Fv))
        r_preds[val].append(r_pred)

    avr_v_preds = {
        freq: torch.stack(v_preds[freq], dim=0).mean(dim=0) 
        for freq in v_preds
    }
    avr_a_preds = {
        freq: torch.stack(a_preds[freq], dim=0).mean(dim=0)
        for freq in a_preds
    }
    avr_r_preds = {
        freq: torch.stack(r_preds[freq], dim=0).mean(dim=0)
        for freq in r_preds
    }

    return avr_v_preds, avr_a_preds, avr_r_preds
