import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''Encoder for the TVAE
    Args:
        input_dim (int):
            入力データの次元数
            前処理で決まる

        hidden_dims (tuple):
            各隠れ層のサイズ
            例: hidden_dims = (128, 128)

        latent_dim (int): 
            エンコーダの埋め込みベクトルの次元数
            潜在空間 (ラテント空間) の次元数．圧縮後のベクトルサイズ
            例: latent_dim = 128
    '''
    def __init__(self, input_dim: int, hidden_dims: tuple, latent_dim: int) -> None:
        super(Encoder, self).__init__()
        dim = input_dim
        sequence = []
        for out_dim in hidden_dims:
            sequence += [nn.Linear(dim, out_dim), nn.ReLU()]
            dim = out_dim

        self.sequence = nn.Sequential(*sequence)
        self.fc_mu = nn.Linear(dim, latent_dim)  # μ
        self.fc_log_var = nn.Linear(dim, latent_dim)  # log(σ^2), nn.Linear(・)->[-∞, ∞]

    def forward(self, x: torch.Tensor):
        h = self.sequence(x)

        mu = self.fc_mu(h)  # μ
        log_var = self.fc_log_var(h)  # log(σ^2)
        std = torch.exp(0.5 * log_var) # σ
        return mu, log_var, std


class Decoder(nn.Module):
    '''Decoder for the TVAE
    Args:
        latent_dim: (int):
            潜在変数の次元数
            例: latent_dim = 128

        hidden_dims (tuple):
            各隠れ層のサイズ
            例: hidden_dims = (128, 128)

        output_dim (int):
            出力データの次元数
    '''
    def __init__(self, latent_dim: int, hidden_dims: tuple, output_dim: int) -> None:
        super(Decoder, self).__init__()
        dim = latent_dim
        sequence = []
        for out_dim in hidden_dims:
            sequence += [nn.Linear(dim, out_dim), nn.ReLU()]
            dim = out_dim
        sequence.append(nn.Linear(dim, output_dim), nn.ReLU())
        
        self.sequence = nn.Sequential(*sequence)
        self.sigma = nn.Parameter(torch.ones(output_dim) * 0.1)  # モデルが確信度を学習するためのパラメータ

    def forward(self, z: torch.Tensor):
        return self.sequence(z), self.sigma


class TVAE(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super(TVAE, self).__init__()
        

    def forward(self, x: torch.Tensor):
        pass

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        pass
