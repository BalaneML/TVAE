from collections import namedtuple

import numpy as np
import pandas as pd
from rdt.transformers import ClusterBasedNormalizer


SpanInfo = namedtuple(
    'SpanInfo',
    ['dim', 'activation_fn']
)

ColumnTransformInfo = namedtuple(
    'ColumTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions']
)


class DataTransformer:
    def __init__(self, max_clusters: int = 10, weitght_threshold: float = 0.005):
        '''
        Args:
            max_clusters (int): ベイズ混合ガウスモデルの最大クラスタ数
            weitght_threshold (float): 分布の重みの閾値 (これ以下の重みを持つ分布は削除される)
        '''
        self.max_clusters = max_clusters
        self.wight_threshold = weitght_threshold

        def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransformInfo:
            '''
            連続値の列をベイズ混合ガウスモデル (Bayesian Gaussian Mixture Model)でフィットする
            Args:
                data (pd.DataFrame): 連続値の列を含むデータフレーム
            Returns:
                ColumnTransformer (namedtuple): 
            '''
            column_name = data.columns[0]

            # ベイズ混合ガウスモデルのインスタンスを作成 -> 連続値列の各ガウス分布の平均，分散，重みを推定
            bayesian_gmm = ClusterBasedNormalizer(
                model_missing_values='from_column',
                max_clusters=min(len(data), self.max_clusters),
                weight_threshold=self.wight_threshold
            )
            bayesian_gmm.fit(data, column_name)
            num_componets = sum(bayesian_gmm.valid_component_indicatior)

            return ColumnTransformInfo(
                column_name=column_name,
                column_type='continuous',
                transform=bayesian_gmm,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_componets, 'softmax')],
                output_dimensions=1+num_componets
            )
