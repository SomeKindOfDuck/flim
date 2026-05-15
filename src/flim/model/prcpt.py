import torch
import torch.distributions as d
import torch.nn as nn
import torch.nn.functional as f

STDNORM = d.Normal(torch.tensor(0.), torch.tensor(1.))


class _SingleVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.visual_in = nn.Linear(input_dim, hidden_dim1)
        self.audio_in = nn.Linear(input_dim, hidden_dim1)
        self.visual_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.audio_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.visual_mu = nn.Linear(hidden_dim2, z_dim)
        self.visual_sigma = nn.Linear(hidden_dim2, z_dim)
        self.audio_mu = nn.Linear(hidden_dim2, z_dim)
        self.audio_sigma = nn.Linear(hidden_dim2, z_dim)

    def reparametarize(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        return mu + sigma.div(2).exp() * epsilon

    def forward(self, visual_in, audio_in):
        visual_h = f.relu(self.visual_in(visual_in))
        audio_h = f.relu(self.audio_in(audio_in))
        visual_h = f.relu(self.visual_hidden(visual_h))
        audio_h = f.relu(self.audio_hidden(audio_h))
        visual_mu = self.visual_mu(visual_h)
        visual_sigma = f.softplus(self.visual_sigma(visual_h))
        audio_mu = self.visual_mu(audio_h)
        audio_sigma = f.softplus(self.visual_sigma(audio_h))
        visual_z = self.reparametarize(visual_mu, visual_sigma)
        audio_z = self.reparametarize(audio_mu, audio_sigma)
        return visual_z, audio_z, visual_mu, visual_sigma, audio_mu, audio_sigma


class _SingleVAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.visual_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.audio_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.visual_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.audio_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.visual_out = nn.Linear(hidden_dim2, input_dim)
        self.audio_out = nn.Linear(hidden_dim2, input_dim)

    def forward(self, z_vis, z_aud):
        visual_h = f.relu(self.visual_hidden1(z_vis))
        audio_h = f.relu(self.audio_hidden1(z_aud))
        visual_h = f.relu(self.visual_hidden2(visual_h))
        audio_h = f.relu(self.audio_hidden2(audio_h))
        visual_out = f.sigmoid(self.visual_out(visual_h))
        audio_out = f.sigmoid(self.audio_out(audio_h))
        return visual_out, audio_out


class SinglwModalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.encoder = _SingleVAEEncoder(input_dim, hidden_dim1, hidden_dim2, z_dim)
        self.decoder = _SingleVAEDecoder(z_dim, hidden_dim2, hidden_dim1, input_dim)

    def forward(self, visual_in, audio_in):
        z_vis, z_aud, mu_vis, sigma_vis, mu_aud, sigma_aud = self.encoder(visual_in, audio_in)
        visual_out, audio_out = self.decoder(z_vis, z_aud)
        return visual_out, audio_out, mu_vis, sigma_vis, mu_aud, sigma_aud, z_vis, z_aud


class _CMVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim, sigma_v, sigma_a):
        super().__init__()
        self.sigma_v = sigma_v
        self.sigma_a = sigma_a
        self.visual_in = nn.Linear(input_dim, hidden_dim1)
        self.audio_in = nn.Linear(input_dim, hidden_dim1)
        self.hidden = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.sigma = nn.Linear(hidden_dim2, z_dim)

    def reparametarize(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        return mu + sigma.div(2).exp() * epsilon

    def forward(self, visual_in, audio_in):
        visual_h = f.relu(self.visual_in(visual_in))
        audio_h = f.relu(self.audio_in(audio_in))
        if self.sigma_v > 0.:
            # visual_h = visual_h * (1 + torch.randn_like(visual_h) * self.sigma_v)
            # visual_h = visual_h + torch.randn_like(visual_h) * (self.sigma_v + visual_h)
            visual_h = visual_h + torch.randn_like(visual_h) * self.sigma_v
        if self.sigma_a > 0.:
            # audio_h = audio_h * (1 + torch.randn_like(audio_h) * self.sigma_a)
            # audio_h = audio_h + torch.randn_like(audio_h) * (self.sigma_a + audio_h)
            audio_h = audio_h + torch.randn_like(audio_h) * self.sigma_a
        cross_h = f.relu(self.hidden(torch.cat((visual_h, audio_h), dim=1)))
        mu = self.mu(cross_h)
        sigma = f.softplus(self.sigma(cross_h))
        z = self.reparametarize(mu, sigma)
        return z, mu, sigma


class _CMVAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.visual_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.audio_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.visual_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.audio_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.visual_out = nn.Linear(hidden_dim2, input_dim)
        self.audio_out = nn.Linear(hidden_dim2, input_dim)

    def forward(self, z):
        visual_h = f.relu(self.visual_hidden1(z))
        audio_h = f.relu(self.audio_hidden1(z))
        visual_h = f.relu(self.visual_hidden2(visual_h))
        audio_h = f.relu(self.audio_hidden2(audio_h))
        visual_out = f.sigmoid(self.visual_out(visual_h))
        audio_out = f.sigmoid(self.audio_out(audio_h))
        return visual_out, audio_out


class CrossModalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim, sigma_v, sigma_a):
        super().__init__()
        self.encoder = _CMVAEEncoder(input_dim, hidden_dim1, hidden_dim2, z_dim, sigma_v, sigma_a)
        self.decoder = _CMVAEDecoder(z_dim, hidden_dim2, hidden_dim1, input_dim)

    def forward(self, visual_in, audio_in):
        z, mu, sigma = self.encoder(visual_in, audio_in)
        visual_out, audio_out = self.decoder(z)
        return visual_out, audio_out, mu, sigma, z


class _LDVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, rnn_dim, z_dim):
        super().__init__()
        self.visual_in = nn.Linear(input_dim, hidden_dim1)
        self.audio_in = nn.Linear(input_dim, hidden_dim1)
        self.hidden = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.rnn = nn.GRU(input_size=hidden_dim2, hidden_size=rnn_dim, num_layers=1, batch_first=True)
        self.mu = nn.Linear(rnn_dim, z_dim)
        self.sigma = nn.Linear(rnn_dim, z_dim)

    def reparametarize(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        return mu + sigma.div(2).exp() * epsilon

    def forward(self, visual_in, audio_in):
        visual_h = f.relu(self.visual_in(visual_in))
        audio_h = f.relu(self.audio_in(audio_in))
        cross_h = f.relu(self.hidden(torch.cat((visual_h, audio_h), dim=1)))
        rnn_h = f.relu(self.rnn(cross_h))
        mu = self.mu(cross_h)
        sigma = f.softplus(self.sigma(cross_h))
        z = self.reparametarize(mu, sigma)
        return z, mu, sigma


class _LDVAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, rnn_dim, input_dim):
        super().__init__()
        self.visual_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.audio_hidden1 = nn.Linear(z_dim, hidden_dim1)
        self.vrnn = nn.GRU(input_size=hidden_dim1, hidden_size=rnn_dim, num_layers=1, batch_first=True)
        self.arnn = nn.GRU(input_size=hidden_dim1, hidden_size=rnn_dim, num_layers=1, batch_first=True)
        self.visual_hidden2 = nn.Linear(rnn_dim, hidden_dim2)
        self.audio_hidden2 = nn.Linear(rnn_dim, hidden_dim2)
        self.visual_out = nn.Linear(hidden_dim2, input_dim)
        self.audio_out = nn.Linear(hidden_dim2, input_dim)

    def forward(self, z):
        visual_h = f.relu(self.visual_hidden1(z))
        audio_h = f.relu(self.audio_hidden1(z))
        visual_h = f.relu(self.vrnn(visual_h))
        audio_h = f.relu(self.arnn(audio_h))
        visual_h = f.relu(self.visual_hidden2(visual_h))
        audio_h = f.relu(self.audio_hidden2(audio_h))
        visual_out = f.sigmoid(self.visual_out(visual_h))
        audio_out = f.sigmoid(self.audio_out(audio_h))
        return visual_out, audio_out


class LatentDynamicVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, rnn_dim, z_dim):
        super().__init__()
        self.encoder = _LDVAEEncoder(input_dim, hidden_dim1, rnn_dim, hidden_dim2, z_dim)
        self.decoder = _LDVAEDecoder(z_dim, hidden_dim2, hidden_dim1, rnn_dim, input_dim)

    def forward(self, visual_in, audio_in):
        z, mu, sigma = self.encoder(visual_in, audio_in)
        visual_out, audio_out = self.decoder(z)
        return visual_out, audio_out, mu, sigma, z


class LatentDynamicPredictor(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.sigma_head = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_seq, beta = 1.):
        rnn_out, _ = self.rnn(z_seq)

        mu = self.mu_head(rnn_out)
        sigma = f.softplus(self.sigma_head(rnn_out))
        epsilon = torch.randn_like(mu)
        z_pred = mu + beta * sigma * epsilon

        return z_pred, mu, sigma


def kl_to_prior_loss(mu, sigma):
    q = d.Normal(mu, sigma.div(2).exp())
    kl_loss = torch.sum(d.kl_divergence(q, STDNORM))
    return kl_loss


def kl_to_cmvae(mu, sigma, mu_pred, sigma_pred):
    q = d.Normal(mu, sigma.div(2).exp())
    q_pred = d.Normal(mu_pred, sigma_pred.div(2).exp())
    return torch.sum(d.kl_divergence(q_pred, q))


def cmvae_reconstruction_loss(visual_pred, audio_pred, visual_obs, audio_obs):
    visual_loss = f.binary_cross_entropy(visual_obs, visual_pred, reduction="sum")
    audio_loss = f.binary_cross_entropy(audio_obs, audio_pred, reduction="sum")
    return visual_loss, audio_loss


def latent_dynamics_loss(z_true, mu, sigma, reduction='sum'):
    pred_dist = d.Normal(mu, sigma + 1e-6)
    nll = -pred_dist.log_prob(z_true).sum()
    return nll
