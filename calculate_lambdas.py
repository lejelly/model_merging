import torch
import os
import json
import numpy as np
from transformers import PreTrainedModel
from proposed_methods import metagpt, WeightingStrategy, metagpt_strict
from typing import List, Optional, Iterator, Tuple, Union
import logging
import pandas as pd
import cProfile
import pstats
from pstats import SortKey
import time


def calculate_and_save_lambdas(
    pretrained_model: PreTrainedModel,
    models_to_merge: List[PreTrainedModel],
    finetuned_model_names: List[str],
    strategy: WeightingStrategy = WeightingStrategy.METAGPT,
    save_dir: Optional[str] = None
) -> np.ndarray:
    """
    選択された戦略に基づいてlambdaを計算し、オプションで結果を保存する関数。
    特定のモデル構成に対応する既存のλ値がある場合は、それを読み込んで返す。
    
    Args:
        pretrained_model: ベースとなる事前学習済みモデル
        models_to_merge: マージ対象のモデルリスト
        finetuned_model_names: モデル名のリスト
        strategy: 使用する計算戦略
        save_dir: 結果を保存するディレクトリパス（Noneの場合は保存しない）
        
    Returns:
        np.ndarray: 計算されたlambda値の配列
    """
    logger = logging.getLogger(__name__)
    
    # 保存ディレクトリが指定されている場合、特定の構成のλ値ファイルを探す
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # モデル構成とストラテジーに基づいてファイル名を生成
        model_names_str = '_'.join(sorted([name.split('/')[-1] for name in finetuned_model_names]))
        lambda_filename = f'lambdas_{strategy.value}_{model_names_str}.csv'
        lambda_filepath = os.path.join(save_dir, lambda_filename)
        
        # 特定の構成のλ値ファイルが存在する場合、読み込む
        if os.path.exists(lambda_filepath):
            logger.info(f"既存のλ値を読み込みます: {lambda_filepath}")
            try:
                lambda_df = pd.read_csv(lambda_filepath)
                # モデル名の順序を合わせる
                lambda_df = lambda_df.set_index('model_name').loc[finetuned_model_names].reset_index()
                return lambda_df['lambda'].values
            except Exception as e:
                logger.warning(f"保存されているλ値の読み込みに失敗しました: {str(e)}")
    
    logger.info(f"Lambda計算戦略: {strategy.value}")
    
    # GPUメモリの初期状態を記録
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # プロファイラの設定
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 実行時間の計測開始
    start_time = time.time()
    
    try:
        # 選択された戦略に基づいてλを計算
        if strategy == WeightingStrategy.METAGPT:
            lambdas = metagpt(pretrained_model, models_to_merge)
        elif strategy == WeightingStrategy.METAGPT_STRICT:
            lambdas = metagpt_strict(pretrained_model, models_to_merge)
        else:
            raise ValueError(f"未対応の戦略です: {strategy.value}")
    finally:
        # プロファイリングの終了
        profiler.disable()
        end_time = time.time()
        
        # GPUメモリ使用量の計算
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        
        # 結果を保存
        if save_dir:
            # λの値をCSVとして保存（固定ファイル名）
            lambda_df = pd.DataFrame({
                'model_name': finetuned_model_names,
                'lambda': lambdas
            })
            lambda_df.to_csv(lambda_filepath, index=False)
            
            # パフォーマンス情報を保存（タイムスタンプ付き）
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            perf_filepath = os.path.join(save_dir, f'performance_{strategy.value}_{model_names_str}_{timestamp}.txt')
            with open(perf_filepath, 'w') as f:
                f.write(f"=== メタGPT実行プロファイル ({strategy.value}) ===\n")
                f.write(f"実行時間: {end_time - start_time:.2f}秒\n")
                f.write(f"GPUメモリ使用量: {memory_used / 1024**2:.2f} MB\n\n")
                
                # プロファイリングの詳細情報を追加
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats(SortKey.TIME)
                stats.print_stats()
            
            logger.info(f"λの計算結果とパフォーマンス情報を保存しました")
            
        # コンソールにも基本的な情報を表示
        print(f"\n=== メタGPT実行プロファイル ({strategy.value}) ===")
        print(f"実行時間: {end_time - start_time:.2f}秒")
        print(f"GPUメモリ使用量: {memory_used / 1024**2:.2f} MB")
        
    return lambdas

