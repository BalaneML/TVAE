import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_transformer import DataTransformer
from src._utils import _format_score


class Encoder(nn.Module):
    '''Encoder for the TVAE
    Args:
        input_dim (int):
            入力データの次元数
            前処理で決まる

        compress_dims (tuple):
            各隠れ層のサイズ
            例: hidden_dims = (128, 128)

        latent_dim (int): 
            エンコーダの埋め込みベクトルの次元数
            潜在空間 (ラテント空間) の次元数．圧縮後のベクトルサイズ
            例: latent_dim = 128
    '''
    def __init__(self, input_dim: int, compress_dims: tuple, latent_dim: int) -> None:
        super(Encoder, self).__init__()
        dim = input_dim
        sequence = []
        for out_dim in compress_dims:
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
    def __init__(self, latent_dim: int, decompress_dims: tuple, output_dim: int) -> None:
        super(Decoder, self).__init__()
        dim = latent_dim
        sequence = []
        for out_dim in decompress_dims:
            sequence += [nn.Linear(dim, out_dim), nn.ReLU()]
            dim = out_dim
        sequence.append(nn.Linear(dim, output_dim))
        
        self.sequence = nn.Sequential(*sequence)
        self.delta = nn.Parameter(torch.ones(output_dim) * 0.1)  # 列ごとの標準偏差 (学習可能のためnn.Parameter)

    def forward(self, z: torch.Tensor):
        return self.sequence(z), self.delta


def _loss_function(recon_x, x, deltas, mu, log_var, output_info, factor):
    '''Loss Function
    Loss = factor * Reconstruction Error (Gaussian NLL, Cross Entropy) + KLD
    '''
    start = 0  # 開始対象列ポインタ
    loss = []

    # 再構成誤差
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':  # 連続列 α, Gaussian NLL
                end = start + span_info.dim  # 最終対象列ポインタ
                std = deltas[start]  # 標準偏差 δ
                eq = x[:, start] - torch.tanh(recon_x[:, start])  # α - tanh(α^-)
                loss.append((eq**2 / (2 * std**2)).sum())  # 二乗誤差項 (ガウス分布の負の対数尤度)
                loss.append(torch.log(std) * x.size()[0])  # 正規化定数 (ガウス分布の負の対数尤度)
                start = end
            else:  # 連続列のモード指定OneHot β or 離散列カテゴリOneHot d, Cross Entropy
                end = start + span_info.dim
                loss.append(
                    F.cross_entropy(recon_x[:, start:end],
                                    torch.argmax(x[:, start:end], dim=1),
                                    reduction='sum'
                                    )  # カテゴリカルクロスエントロピー
                )
                start = end
    assert start == recon_x.size()[1]

    # KLダイバージェンス 
    kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    return sum(loss) * factor / x.size()[0], kl / x.size()[0]  # sum(loss) = -logp(α)=(α-α^-)^2 / 2δ^2 + logδ, x.size()[0]->バッチ平均 


class TVAE:
    '''TVAE'''
    def __init__(
            self,
            latent_dim: int=128,
            compress_dims: tuple=(128, 128),
            decompress_dims: tuple=(128, 128),

            l2scale=1e-5,
            batch_size=500,
            epochs=300,
            loss_factor=2,
            verbose=False,
            device='cpu'
            ) -> None:
        self.latent_dim = latent_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.verbose = verbose
        self._device = device
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        
    def fit(self, train_data, discrete_columns) -> None:
        '''Fit the TVAE
        Args:
            train_data (numpy.ndarry or pandas.DataFrame):

            discrete_columns (list_like):
        '''
        # data transfomer, loader
        self.data_transformer = DataTransformer()
        self.data_transformer.fit(train_data, discrete_columns)
        train_data = self.data_transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Encoder, Decoder
        data_dim = self.data_transformer.output_dimensions
        self.encoder = Encoder(data_dim, self.compress_dims, self.latent_dim).to(self._device)
        self.decoder = Decoder(self.latent_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )

        # 学習プロセス可視化
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {Loss}'
            iterator.set_description(iterator_description.format(Loss=_format_score(0)))

        # 学習ループ
        self.encoder.train()
        self.decoder.train()
        for i in iterator:
            loss_value = []
            batch = []

            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()  # 勾配リセット
                real = data[0].to(self._device)

                mu, log_var, std = self.encoder(real)  # Encoder(x) -> μ, log(σ^2), σ
                eps = torch.randn_like(std)  # ε~N(0,I)
                z = mu + std * eps  #z ~ N(μ, σ²) = μ + σ * ε，再パラメータ化トリック
                recon_x, deltas = self.decoder(z)  

                # 損失計算
                recon_loss, kl_loss=_loss_function(
                    recon_x,
                    real,
                    deltas,
                    mu,
                    log_var,
                    self.data_transformer.output_info_list,
                    self.loss_factor
                )
                loss = recon_loss + kl_loss  #  バッチの損失，ELBOの最小化
                
                loss.backward()  # 逆伝播で勾配計算
                optimizerAE.step()  # パラメータ更新
                self.decoder.delta.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_value.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_value
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df
            
            if self.verbose:  # 学習ログ
                iterator.set_description(
                    iterator_description.format(Loss=_format_score(loss.detach().cpu().item()))
                )

    def sample(self, samples):
        '''Sample data from The trained TVAE
        Args:
            samples (int):
                Number of rows to sample
        Return:

        '''
        self.decoder.eval()
        steps = samples // self.batch_size + 1
        data = []

        with torch.no_grad():
            for _ in range(steps):
                mean = torch.zeros(self.batch_size, self.latent_dim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self._device)
                fake, deltas = self.decoder(noise)
                fake = torch.tanh(fake)
                data.append(fake.cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = data[:samples]
        self.decoder.train()
        return self.data_transformer.inverse_transform(data, deltas.detach().cpu().numpy())
