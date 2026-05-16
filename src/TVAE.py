import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # r_j -> 128
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 128 -> 128

        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)  # 128 -> 128
        self.fc_log_var = nn.Linear(hidden_dim, hidden_dim)  # 128 -> 128

    def forward(self, x: torch.Tensor):
        # 全結合層
        h = F.relu(self.fc1(x))  # r_j -> 128
        h = F.relu(self.fc2(h))  # 128 -> 128

        # 平均と分散の計算
        mu = self.fc_mu(h)  # μ: 128 -> 128
        log_var = self.fc_log_var(h)  # log(σ^2): 128 -> 128

        # 潜在変数を求める (reparameterization trick)
        eps = torch.randn_like(log_var)  # 標準正規分布からサンプリング
        z = mu + torch.exp(0.5 * log_var) * eps  # 潜在変数 z = μ + σ * ε
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim: int=128, hidden_dim: int=128, latent_dim: int=128, m_i: int=10, D_i: int=10) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # z -> 128
        self.fc2 = nn.Linear(hidden_dim, latent_dim)  # 128 -> 128

        self.fc_alpha = nn.Linear(latent_dim, 1)  # α: 128 -> 1 連続値
        self.fc_beta = nn.Linear(latent_dim, m_i)  # β:　128 -> m_i モード
        self.fc_d = nn.Linear(latent_dim, D_i)  # D: 128 -> D_i 離散値

    def forward(self, z: torch.Tensor):
        # 全結合層
        h = F.relu(self.fc1(z))  # z -> 128
        h = F.relu(self.fc2(h))  # 128 -> 128

        # αの計算
        alpha = self.fc_alpha(h)  # α: 128 -> 1
        alpha = F.tanh(alpha)  # αを-1から1の範囲に制限

        # βの計算
        beta = self.fc_beta(h)  # β: 128 -> m_i
        beta = F.softmax(beta, dim=-1)

        # Dの計算
        d = self.fc_d(h)  # D: 128 -> D_i
        d = F.softmax(d, dim=-1)

        output = alpha * beta * d  # 出力の組み合わせ
        return output, alpha, beta, d


class TVAE(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super(TVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        z, mu, log_var = self.encoder(x)
        output, alpha, beta, d = self.decoder(z)
        return output, alpha, beta, d, mu, log_var

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # 再構築誤差
        recon_loss = F.cross_entropy(recon_x, x)

        # KLダイバージェンス
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_loss
