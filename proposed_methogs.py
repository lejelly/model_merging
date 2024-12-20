from enum import Enum
from typing import List, Optional, Iterator, Tuple, Union
import torch
import numpy as np
from torch import nn
from transformers import PreTrainedModel
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

class WeightingStrategy(Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    GRAPH_LAPLACIAN = "graph_laplacian"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    MULTITASK_LEARNING = "multitask_learning" 
    ATTENTION_BASED = "attention_based"

class TaskVectorCalculator:
    @staticmethod
    def parameter_iterator(
        pretrained_model: PreTrainedModel,
        model: PreTrainedModel
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """メモリ効率の良いパラメータイテレータ"""
        for (_, param_0), (_, param_t) in zip(
            pretrained_model.named_parameters(),
            model.named_parameters()
        ):
            yield param_0, param_t

    @staticmethod
    def calculate_squared_norm(
        pretrained_model: PreTrainedModel,
        model: PreTrainedModel
    ) -> float:
        """メモリ効率の良いノルム計算"""
        squared_norm = 0.0
        for param_0, param_t in TaskVectorCalculator.parameter_iterator(pretrained_model, model):
            param_0_cpu = param_0.detach().cpu().to(torch.float32)
            param_t_cpu = param_t.detach().cpu().to(torch.float32)
            diff = param_t_cpu - param_0_cpu
            squared_norm += torch.sum(diff * diff).item()
            del param_0_cpu, param_t_cpu, diff
            torch.cuda.empty_cache()
        return squared_norm

    @staticmethod
    def calculate_cosine_similarity(
        pretrained_model: PreTrainedModel,
        model_i: PreTrainedModel,
        model_j: PreTrainedModel
    ) -> float:
        """メモリ効率の良いコサイン類似度計算"""
        dot_product = 0.0
        norm_i = 0.0
        norm_j = 0.0
        
        for (param_0, param_i), (_, param_j) in zip(
            TaskVectorCalculator.parameter_iterator(pretrained_model, model_i),
            model_j.named_parameters()
        ):
            param_0_cpu = param_0.detach().cpu().to(torch.float32)
            param_i_cpu = param_i.detach().cpu().to(torch.float32)
            param_j_cpu = param_j.detach().cpu().to(torch.float32)
            
            diff_i = param_i_cpu - param_0_cpu
            diff_j = param_j_cpu - param_0_cpu
            
            dot_product += torch.sum(diff_i * diff_j).item()
            norm_i += torch.sum(diff_i * diff_i).item()
            norm_j += torch.sum(diff_j * diff_j).item()
            
            del param_0_cpu, param_i_cpu, param_j_cpu, diff_i, diff_j
            torch.cuda.empty_cache()
        
        return dot_product / (np.sqrt(norm_i) * np.sqrt(norm_j))

class WeightAdjuster:
    @staticmethod
    def calculate_similarity_matrix(
        pretrained_model: PreTrainedModel,
        models_to_merge: List[PreTrainedModel],
        batch_size: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """メモリ効率の良い類似度行列計算"""
        n_models = len(models_to_merge)
        similarity_matrix = np.zeros((n_models, n_models))
        
        # ノルムの計算
        norms = []
        denominator = 0.0
        for i in tqdm(range(n_models), desc="Calculating norms"):
            squared_norm = TaskVectorCalculator.calculate_squared_norm(
                pretrained_model, models_to_merge[i]
            )
            norms.append(squared_norm)
            denominator += squared_norm
        
        base_lambdas = np.array(norms) / denominator
        
        # バッチ処理で類似度を計算
        for i in tqdm(range(0, n_models, batch_size), desc="Computing similarities"):
            batch_i = min(i + batch_size, n_models)
            for j in range(i, n_models, batch_size):
                batch_j = min(j + batch_size, n_models)
                
                for idx_i in range(i, batch_i):
                    for idx_j in range(max(j, idx_i + 1), batch_j):
                        cos_sim = TaskVectorCalculator.calculate_cosine_similarity(
                            pretrained_model,
                            models_to_merge[idx_i],
                            models_to_merge[idx_j]
                        )
                        similarity_matrix[idx_i, idx_j] = cos_sim
                        similarity_matrix[idx_j, idx_i] = cos_sim
        
        return similarity_matrix, base_lambdas

    @staticmethod
    def cosine_similarity_adjustment(
        similarity_matrix: np.ndarray,
        base_lambdas: np.ndarray
    ) -> np.ndarray:
        """コサイン類似度ベースの調整"""
        n_models = len(base_lambdas)
        alpha_adjustments = []
        
        for i in range(n_models):
            similarities = [
                1 - similarity_matrix[i, j]
                for j in range(n_models)
                if i != j
            ]
            alpha = np.mean(similarities)
            alpha_adjustments.append(alpha)
        
        alpha_adjustments = np.array(alpha_adjustments)
        alpha_adjustments = alpha_adjustments / np.sum(alpha_adjustments)
        return base_lambdas * alpha_adjustments

    @staticmethod
    def graph_laplacian_adjustment(
        similarity_matrix: np.ndarray,
        base_lambdas: np.ndarray,
        mu: float = 0.1
    ) -> np.ndarray:
        """グラフラプラシアンベースの調整"""
        n_models = len(base_lambdas)
        D = np.diag(np.sum(similarity_matrix, axis=1))
        L = D - similarity_matrix
        
        I = np.eye(n_models)
        lambdas = np.linalg.solve(L + mu * I, mu * base_lambdas)
        return lambdas / np.sum(lambdas)

    @staticmethod
    def hierarchical_clustering_adjustment(
        similarity_matrix: np.ndarray,
        base_lambdas: np.ndarray,
        n_clusters: int = 2
    ) -> np.ndarray:
        """階層的クラスタリングベースの調整"""
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        linkage = hierarchy.linkage(squareform(distance_matrix), method='ward')
        clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        
        adjusted_lambdas = base_lambdas.copy()
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = clusters == cluster_id
            cluster_size = np.sum(cluster_mask)
            if cluster_size > 1:
                adjusted_lambdas[cluster_mask] *= (1 + 1/cluster_size)
        
        return adjusted_lambdas / np.sum(adjusted_lambdas)

    @staticmethod
    def multitask_learning_adjustment(
        pretrained_model: PreTrainedModel,
        models_to_merge: List[PreTrainedModel],
        base_lambdas: np.ndarray,
        gamma: float = 0.1,
        batch_size: int = 2
    ) -> np.ndarray:
        """メモリ効率化されたマルチタスク学習ベースの調整
        
        Args:
            pretrained_model: 基準となるモデル
            models_to_merge: マージするモデルのリスト
            base_lambdas: 基本的な重み係数
            gamma: 干渉の制御パラメータ
            batch_size: バッチサイズ
            
        Returns:
            調整された重み係数
        """
        n_models = len(models_to_merge)
        beta_adjustments = []
        
        # ノルムの計算（再利用のため）
        norms = []
        for model in tqdm(models_to_merge, desc="Calculating norms"):
            squared_norm = TaskVectorCalculator.calculate_squared_norm(
                pretrained_model, model
            )
            norms.append(np.sqrt(squared_norm))
        
        # バッチ処理でβを計算
        for i in tqdm(range(0, n_models, batch_size), desc="Computing beta adjustments"):
            batch_i = min(i + batch_size, n_models)
            
            # 現在のバッチのモデルに対するβを計算
            for idx_i in range(i, batch_i):
                interference_sum = 0.0
                
                # 他のすべてのモデルとの干渉を計算
                for idx_j in range(n_models):
                    if idx_i != idx_j:
                        # 内積を効率的に計算
                        dot_product = 0.0
                        for (param_0, param_i), (_, param_j) in zip(
                            TaskVectorCalculator.parameter_iterator(
                                pretrained_model, models_to_merge[idx_i]
                            ),
                            models_to_merge[idx_j].named_parameters()
                        ):
                            param_0_cpu = param_0.detach().cpu().to(torch.float32)
                            param_i_cpu = param_i.detach().cpu().to(torch.float32)
                            param_j_cpu = param_j.detach().cpu().to(torch.float32)
                            
                            diff_i = param_i_cpu - param_0_cpu
                            diff_j = param_j_cpu - param_0_cpu
                            
                            dot_product += torch.sum(diff_i * diff_j).item()
                            
                            del param_0_cpu, param_i_cpu, param_j_cpu, diff_i, diff_j
                            torch.cuda.empty_cache()
                        
                        # 正規化された内積の絶対値を計算
                        normalized_interference = abs(dot_product) / (norms[idx_i] * norms[idx_j])
                        interference_sum += normalized_interference
                
                # βの計算
                beta = np.exp(-gamma * interference_sum)
                beta_adjustments.append(beta)
        
        # 配列に変換して正規化
        beta_adjustments = np.array(beta_adjustments)
        beta_adjustments = beta_adjustments / np.sum(beta_adjustments)
        
        # 最終的な重み係数の計算
        final_lambdas = base_lambdas * beta_adjustments
        final_lambdas = final_lambdas / np.sum(final_lambdas)
        
        return final_lambdas

    @staticmethod
    def attention_based_adjustment(
        similarity_matrix: np.ndarray,
        base_lambdas: np.ndarray,
        pretrained_model: PreTrainedModel,
        models_to_merge: List[PreTrainedModel]
    ) -> np.ndarray:
        """メモリ効率化された注意機構ベースの調整"""
        n_models = len(models_to_merge)
        
        # 注意スコアをバッチで計算
        attention_scores = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    # 内積を効率的に計算
                    dot_product = 0.0
                    param_count = 0
                    for (param_0, param_i), (_, param_j) in zip(
                        TaskVectorCalculator.parameter_iterator(pretrained_model, models_to_merge[i]),
                        models_to_merge[j].named_parameters()
                    ):
                        param_0_cpu = param_0.detach().cpu().to(torch.float32)
                        param_i_cpu = param_i.detach().cpu().to(torch.float32)
                        param_j_cpu = param_j.detach().cpu().to(torch.float32)
                        
                        diff_i = param_i_cpu - param_0_cpu
                        diff_j = param_j_cpu - param_0_cpu
                        
                        dot_product += torch.sum(diff_i * diff_j).item()
                        param_count += diff_i.numel()
                        
                        del param_0_cpu, param_i_cpu, param_j_cpu, diff_i, diff_j
                        torch.cuda.empty_cache()
                    
                    # スケーリングされた注意スコアを計算
                    attention_scores[i, j] = dot_product / np.sqrt(param_count)
        
        # Softmax適用
        attention_weights = np.exp(attention_scores)
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # 最終的な重み計算
        attention_adjustment = np.sum(attention_weights, axis=1)
        attention_adjustment = attention_adjustment / np.sum(attention_adjustment)
        
        return base_lambdas * attention_adjustment

def metagpt_advanced(
    pretrained_model: PreTrainedModel,
    models_to_merge: List[PreTrainedModel],
    strategy: WeightingStrategy,
    strategy_params: Optional[dict] = None,
    batch_size: int = 2
) -> np.ndarray:
    """メモリ最適化された拡張版metagpt関数"""
    if strategy_params is None:
        strategy_params = {}
    
    # 類似度行列とbase_lambdasの計算
    similarity_matrix, base_lambdas = WeightAdjuster.calculate_similarity_matrix(
        pretrained_model, models_to_merge, batch_size
    )
    
    # 選択された戦略に基づく重み調整
    if strategy == WeightingStrategy.COSINE_SIMILARITY:
        final_lambdas = WeightAdjuster.cosine_similarity_adjustment(
            similarity_matrix, base_lambdas
        )
    elif strategy == WeightingStrategy.GRAPH_LAPLACIAN:
        mu = strategy_params.get('mu', 0.1)
        final_lambdas = WeightAdjuster.graph_laplacian_adjustment(
            similarity_matrix, base_lambdas, mu
        )
    elif strategy == WeightingStrategy.HIERARCHICAL_CLUSTERING:
        n_clusters = strategy_params.get('n_clusters', 2)
        final_lambdas = WeightAdjuster.hierarchical_clustering_adjustment(
            similarity_matrix, base_lambdas, n_clusters
        )
    elif strategy == WeightingStrategy.MULTITASK_LEARNING:
        gamma = strategy_params.get('gamma', 0.1)
        final_lambdas = WeightAdjuster.multitask_learning_adjustment(
            pretrained_model,
            models_to_merge,
            base_lambdas,
            gamma=gamma,
            batch_size=batch_size
        )
    elif strategy == WeightingStrategy.ATTENTION_BASED:
        final_lambdas = WeightAdjuster.attention_based_adjustment(
            similarity_matrix, 
            base_lambdas,
            pretrained_model,
            models_to_merge
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return final_lambdas

class TaskVectorVisualizer:
    """タスクベクトルの可視化を行うクラス"""
    
    def __init__(
        self,
        pretrained_model: PreTrainedModel,
        models: List[PreTrainedModel],
        model_names: Optional[List[str]] = None
    ):
        self.pretrained_model = pretrained_model
        self.models = models
        self.n_models = len(models)
        
        if model_names is None:
            self.model_names = [f"Model_{i}" for i in range(self.n_models)]
        else:
            if len(model_names) != self.n_models:
                raise ValueError("model_names length must match the number of models")
            self.model_names = model_names
            
        self.similarity_matrix = self._calculate_similarity_matrix()
        
    def _calculate_similarity_matrix(self) -> np.ndarray:
        """モデル間の類似度行列を計算"""
        matrix = np.zeros((self.n_models, self.n_models))
        
        for i in tqdm(range(self.n_models), desc="Computing similarity matrix"):
            for j in range(i, self.n_models):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    similarity = TaskVectorCalculator.calculate_cosine_similarity(
                        self.pretrained_model,
                        self.models[i],
                        self.models[j]
                    )
                    matrix[i, j] = similarity
                    matrix[j, i] = similarity
        
        return matrix
    
    def plot_confusion_matrix(
        self,
        figsize: tuple = (10, 8),
        cmap: str = "YlOrRd",
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        fmt: str = ".2f"
    ) -> None:
        """混同行列をヒートマップとして可視化"""
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            self.similarity_matrix,
            annot=True,
            cmap=cmap,
            fmt=fmt,
            square=True,
            xticklabels=self.model_names,
            yticklabels=self.model_names,
            vmin=-1,
            vmax=1,
            center=0
        )
        
        plt.title("Task Vector Cosine Similarity Matrix")
        plt.tight_layout()
        
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def save_csv(self, save_path: Union[str, Path]) -> None:
        """混同行列をCSVファイルとして保存"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(
            self.similarity_matrix,
            index=self.model_names,
            columns=self.model_names
        )
        df.to_csv(save_path)

def visualize_task_similarities(
    pretrained_model: PreTrainedModel,
    models: List[PreTrainedModel],
    model_names: Optional[List[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    file_prefix: str = "task_vector",
    figsize: tuple = (10, 8),
    dpi: int = 300
) -> TaskVectorVisualizer:
    """
    タスクベクトル間の類似度を計算し、混同行列として可視化・保存する関数
    
    Args:
        pretrained_model: ベースとなる事前学習済みモデル
        models: 比較する fine-tuned モデルのリスト
        model_names: モデルの名前リスト
        save_dir: 保存先ディレクトリ（Noneの場合は保存しない）
        file_prefix: 保存するファイルの接頭辞
        figsize: 図のサイズ
        dpi: 保存時の解像度
        
    Returns:
        TaskVectorVisualizer インスタンス
    """
    visualizer = TaskVectorVisualizer(
        pretrained_model=pretrained_model,
        models=models,
        model_names=model_names
    )
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像として保存
        visualizer.plot_confusion_matrix(
            figsize=figsize,
            save_path=save_dir / f"{file_prefix}_similarity.png",
            dpi=dpi
        )
        
        # CSVとして保存
        visualizer.save_csv(save_dir / f"{file_prefix}_similarity.csv")
    else:
        visualizer.plot_confusion_matrix(figsize=figsize)
    
    return visualizer

def calculate_lambda_optimized(theta_0_model, theta_t_models, sample_ratio=0.01, random_state=42):
    """
    RAM使用量を最適化した重み係数λを計算する関数

    Parameters:
    - theta_0_model: 事前学習後のモデルオブジェクト
    - theta_t_models: ファインチューニング後のモデルオブジェクトのリスト
    - sample_ratio: サンプリングするパラメータの割合（0 < sample_ratio <= 1）
    - random_state: ランダムシード

    Returns:
    - lambda_list: 重み係数のリスト
    """
    import torch
    from scipy.spatial.distance import pdist, squareform
    import umap
    import numpy as np

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # モデルのパラメータを取得
    theta_0_state_dict = theta_0_model.state_dict()
    param_keys = list(theta_0_state_dict.keys())
    num_params = len(param_keys)
    
    # サンプリングするインデックスを取得
    sample_size = int(num_params * sample_ratio)
    sample_indices = np.random.choice(num_params, size=sample_size, replace=False)
    sample_indices.sort()
    
    # タスク数を取得
    num_tasks = len(theta_t_models)

    # タスクベクトルを計算（サンプリングしたパラメータのみ）
    task_vectors = []
    for t, theta_t_model in enumerate(theta_t_models):
        delta_params = []
        theta_t_state_dict = theta_t_model.state_dict()
        for idx in sample_indices:
            key = param_keys[idx]
            # パラメータを取得しフラット化
            theta_0_param = theta_0_state_dict[key].detach().cpu().to(torch.float32).numpy().flatten()
            theta_t_param = theta_t_state_dict[key].detach().cpu().to(torch.float32).numpy().flatten()
            # 差分を計算
            delta = theta_t_param - theta_0_param
            # 差分をdelta_paramsに追加
            delta_params.extend(delta)
        # タスクベクトルに追加
        task_vectors.append(np.array(delta_params))

    # ベクトルを配列に変換
    X = np.array(task_vectors)  # Xの形状は (num_tasks, vector_length)

    # UMAPの適用（random_stateを固定）
    reducer = umap.UMAP(
        n_neighbors=5,          # データポイントが増えるので適宜調整
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=random_state,
        low_memory=True,
        init='random'           # スペクトラル埋め込みを無効化
    )
    embedding = reducer.fit_transform(X)

    # タスクベクトル間の距離を計算
    distances = squareform(pdist(embedding, metric='euclidean'))

    # タスクごとの平均距離を計算
    D_list = []
    for i in range(num_tasks):
        sum_distances = np.sum(distances[i]) - distances[i][i]  # 自分自身との距離を除く
        mean_distance = sum_distances / (num_tasks - 1)
        D_list.append(mean_distance)

    # 平均距離を正規化して重みを計算
    D_total = np.sum(D_list)
    lambda_prime = [D / D_total for D in D_list]

    # 重みに1を加算
    lambda_list = [lp + 1 for lp in lambda_prime]
    fig = visualize_umap_embedding(embedding, lambda_prime)
    fig.savefig('/work/gb20/b20042/model_merging/figs/math_code_jp/task_visualization.png', dpi=300, bbox_inches='tight')

    return lambda_list

def visualize_umap_embedding(embedding, lambda_list=None, figsize=(10, 8)):
    """
    UMAPの埋め込み結果を可視化する関数
    
    Parameters:
    - embedding: UMAP embedddingの結果 (n_tasks × 2の配列)
    - lambda_list: 各タスクの重み係数（オプション）
    - figsize: 図のサイズ
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # プロットの準備
    plt.figure(figsize=figsize)
    
    # 散布図のプロット
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=lambda_list if lambda_list is not None else None,
        cmap='viridis',
        s=100,
    )
    
    # タスク番号のラベル付け
    for i in range(len(embedding)):
        plt.annotate(
            f'Task {i+1}',
            (embedding[i, 0], embedding[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )
    
    # λ値がある場合はカラーバーを追加
    if lambda_list is not None:
        plt.colorbar(scatter, label='λ value')
    
    # グラフの設定
    plt.title('UMAP Embedding of Task Vectors')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, alpha=0.3)
    
    # 縦軸と横軸のスケールを合わせる
    plt.axis('equal')
    
    # タイトな配置に調整
    plt.tight_layout()
    
    return plt.gcf()

def calculate_lambda_full(theta_0_model, theta_t_models, random_state=42):
    """
    フルパラメータを使用して重み係数λを計算する関数
    Parameters:
    - theta_0_model: 事前学習後のモデルオブジェクト
    - theta_t_models: ファインチューニング後のモデルオブジェクトのリスト
    - random_state: ランダムシード
    Returns:
    - lambda_list: 重み係数のリスト
    """
    import torch
    from scipy.spatial.distance import pdist, squareform
    import umap
    import numpy as np
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # モデルのパラメータを取得
    theta_0_state_dict = theta_0_model.state_dict()
    
    # タスク数を取得
    num_tasks = len(theta_t_models)
    # タスクベクトルを計算（全パラメータ）
    task_vectors = []
    for t, theta_t_model in enumerate(theta_t_models):
        delta_params = []
        theta_t_state_dict = theta_t_model.state_dict()
        for key in theta_0_state_dict.keys():
            # パラメータを取得しフラット化
            theta_0_param = theta_0_state_dict[key].detach().cpu().to(torch.float32).numpy().flatten()
            theta_t_param = theta_t_state_dict[key].detach().cpu().to(torch.float32).numpy().flatten()
            # 差分を計算
            delta = theta_t_param - theta_0_param
            # 差分をdelta_paramsに追加
            delta_params.extend(delta)
        # タスクベクトルに追加
        task_vectors.append(np.array(delta_params))
    # ベクトルを配列に変換
    X = np.array(task_vectors)  # Xの形状は (num_tasks, vector_length)
    
    # UMAPの適用（random_stateを固定）
    reducer = umap.UMAP(
        n_neighbors=5,          # データポイントが増えるので適宜調整
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=random_state,
        low_memory=True,
        init='random'           # スペクトラル埋め込みを無効化
    )
    embedding = reducer.fit_transform(X)
    
    # タスクベクトル間の距離を計算
    distances = squareform(pdist(embedding, metric='euclidean'))
    # タスクごとの平均距離を計算
    D_list = []
    for i in range(num_tasks):
        sum_distances = np.sum(distances[i]) - distances[i][i]  # 自分自身との距離を除く
        mean_distance = sum_distances / (num_tasks - 1)
        D_list.append(mean_distance)
    
    # 平均距離を正規化して重みを計算
    D_total = np.sum(D_list)
    lambda_prime = [D / D_total for D in D_list]
    # 重みに1を加算
    lambda_list = [lp + 1 for lp in lambda_prime]
    
    fig = visualize_umap_embedding(embedding, lambda_prime)
    fig.savefig('/work/gb20/b20042/model_merging/figs/math_code_jp/task_visualization.png', dpi=300, bbox_inches='tight')
    
    return lambda_list

def analyze_task_vectors(
    pretrained_model: PreTrainedModel,
    models_to_merge: List[PreTrainedModel]
) -> dict:
    """
    タスクベクトル間のグラム行列を計算し、その性質を分析する

    Args:
        pretrained_model: 基準となる事前学習済みモデル
        models_to_merge: マージするモデルのリスト

    Returns:
        dict: 分析結果を含む辞書
        {
            "gram_matrix": グラム行列,
            "has_inverse": 逆行列が存在するかどうか,
            "condition_number": 条件数,
            "determinant": 行列式
        }
    """
    n_models = len(models_to_merge)
    gram_matrix = np.zeros((n_models, n_models))
    
    # グラム行列の計算
    for i in tqdm(range(n_models), desc="Computing Gram matrix"):
        for j in range(i, n_models):
            if i == j:
                # 対角成分は二乗ノルム
                squared_norm = TaskVectorCalculator.calculate_squared_norm(
                    pretrained_model, models_to_merge[i]
                )
                gram_matrix[i, j] = squared_norm
            else:
                # 非対角成分は内積
                dot_product = 0.0
                for (param_0, param_i), (_, param_j) in zip(
                    TaskVectorCalculator.parameter_iterator(pretrained_model, models_to_merge[i]),
                    models_to_merge[j].named_parameters()
                ):
                    param_0_cpu = param_0.detach().cpu().to(torch.float32)
                    param_i_cpu = param_i.detach().cpu().to(torch.float32)
                    param_j_cpu = param_j.detach().cpu().to(torch.float32)
                    
                    diff_i = param_i_cpu - param_0_cpu
                    diff_j = param_j_cpu - param_0_cpu
                    
                    dot_product += torch.sum(diff_i * diff_j).item()
                    
                    del param_0_cpu, param_i_cpu, param_j_cpu, diff_i, diff_j
                    torch.cuda.empty_cache()
                
                gram_matrix[i, j] = dot_product
                gram_matrix[j, i] = dot_product
    
    # 行列式の計算
    det = np.linalg.det(gram_matrix)
    
    # 条件数の計算
    condition_number = np.linalg.cond(gram_matrix)
    
    # 逆行列の存在判定
    # 行列式がゼロに近い、または条件数が大きすぎる場合は逆行列が不安定
    has_inverse = abs(det) > 1e-10 and condition_number < 1e15
    
    return {
        "gram_matrix": gram_matrix,
        "has_inverse": has_inverse,
        "condition_number": condition_number,
        "determinant": det
    }
    
def calculate_optimal_lambdas(pretrained_model, models_to_merge):
    """
    タスクベクトル間の関係を考慮して最適なλを計算する関数
    
    Args:
        pretrained_model: 事前学習済みモデル (θ₀)
        models_to_merge: ファインチューニング済みモデルのリスト (θₖ)
    
    Returns:
        最適化されたλのリスト
    """
    n_models = len(models_to_merge)
    lambdas = np.zeros(n_models)
    
    # グラム行列の計算
    gram_matrix = np.zeros((n_models, n_models))
    norms = np.zeros(n_models)
    
    for i in range(n_models):
        # ||θₜ - θ₀||²の計算
        norms[i] = TaskVectorCalculator.calculate_squared_norm(
            pretrained_model, models_to_merge[i]
        )
        
        for j in range(i, n_models):
            if i == j:
                gram_matrix[i, j] = norms[i]
            else:
                # (θₖ - θ₀)(θₜ - θ₀)ᵀの計算
                dot_product = 0.0
                for (param_0, param_i), (_, param_j) in zip(
                    TaskVectorCalculator.parameter_iterator(pretrained_model, models_to_merge[i]),
                    models_to_merge[j].named_parameters()
                ):
                    param_0_cpu = param_0.detach().cpu().to(torch.float32)
                    param_i_cpu = param_i.detach().cpu().to(torch.float32)
                    param_j_cpu = param_j.detach().cpu().to(torch.float32)
                    
                    diff_i = param_i_cpu - param_0_cpu
                    diff_j = param_j_cpu - param_0_cpu
                    
                    dot_product += torch.sum(diff_i * diff_j).item()
                    
                    del param_0_cpu, param_i_cpu, param_j_cpu, diff_i, diff_j
                    torch.cuda.empty_cache()
                
                gram_matrix[i, j] = dot_product
                gram_matrix[j, i] = dot_product
    
    # λの計算
    for t in range(n_models):
        numerator = 0.0
        for k in range(n_models):
            if k != t:
                numerator += lambdas[k] * gram_matrix[k, t]
        
        lambdas[t] = 1.0 - numerator / norms[t]
    
    return lambdas