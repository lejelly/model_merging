import torch
from optimize_lambdas_cross_entropy import LambdaOptimizerCrossEntropy, LayerwiseMergedModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random

def test_lambda_optimization():
    """λの最適化プロセスをテストする"""
    # メモリ使用量を削減するための修正
    device = "cuda"
    torch.cuda.empty_cache()  # キャッシュをクリア
    
    # 1. 必要なモデルとトークナイザーの準備
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/open-calm-small",
        torch_dtype=torch.float16  # 半精度で読み込み
    ).to(device)
    tokenizers = [
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small"),
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small"),
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small")
    ]
    
    # モデルを3つ用意し、それぞれ異なる係数でパラメータをずらす
    models_to_merge = []
    for i, scale in enumerate([1.2, 0.8, 1.5]):  # 異なるスケール係数
        model = AutoModelForCausalLM.from_pretrained(
            "cyberagent/open-calm-small",
            torch_dtype=torch.float16
        ).to(device)
        # パラメータを人為的にスケーリング
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
        models_to_merge.append(model)
        print(f"Model {i+1} scaled by {scale}")
        
    # 不要なモデルは明示的に解放
    del model
    torch.cuda.empty_cache()

    # 2. オプティマイザーの初期化
    optimizer = LambdaOptimizerCrossEntropy(
        seed=42,
        pretrained_model=pretrained_model,
        pretrained_model_name="cyberagent/open-calm-small",
        tokenizers=tokenizers,
        models_to_merge=models_to_merge,
        initial_lambdas=[0.1, 0.1, 0.1],
        num_epochs=2,
        batch_size=2,
        num_train_samples=2
    )

    print("\n=== λの最適化テスト ===")
    
    # 初期状態を保存
    initial_lambdas = optimizer.lambdas.clone().detach()
    print("初期λ値:", initial_lambdas.cpu().numpy())
    print("λのrequires_grad:", optimizer.lambdas.requires_grad)
    
    # 最適化を実行
    print("\n最適化実行中...")
    
    # 1エポック目の勾配を確認
    loss = optimizer.train_one_epoch(batch_size=2)
    print("\n1エポック目の結果:")
    print(f"Loss: {loss}")
    print(f"λの勾配: {optimizer.lambdas.grad}")
    print(f"現在のλ値: {optimizer.lambdas.detach().cpu().numpy()}")
    
    # 完全な最適化を実行
    optimized_lambdas = optimizer.optimize()
    
    print("\n最終的な最適化結果:")
    print("最適化後のλ値:", optimized_lambdas)
    print("λの変化量:", optimized_lambdas - initial_lambdas.cpu().numpy())
    
    # 検証
    assert not np.allclose(optimized_lambdas, initial_lambdas.cpu().numpy()), \
        "λ値が更新されていません"
    
    assert len(set(optimized_lambdas)) > 1, \
        "すべてのλが同じ値になっています"
    
    print("\n✓ λの最適化テスト成功")
    return True

if __name__ == "__main__":    
    # 新しい最適化テストを実行
    optimization_test_success = test_lambda_optimization()
    print(f"\n最適化テスト: {'成功' if optimization_test_success else '失敗'}!")