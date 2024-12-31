import torch
from optimize_lambdas_cross_entropy import LambdaOptimizerCrossEntropy
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_lambda_gradients():
    # 1. 必要なモデルとトークナイザーの準備
    pretrained_model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small")
    tokenizers = [
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small"),
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small"),
        AutoTokenizer.from_pretrained("cyberagent/open-calm-small")
    ]
    
    # テスト用の小さなモデルを3つ用意
    models_to_merge = [
        AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small"),
        AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small"),
        AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small")
    ]

    # 2. オプティマイザーの初期化
    optimizer = LambdaOptimizerCrossEntropy(
        seed=42,
        pretrained_model=pretrained_model,
        pretrained_model_name="cyberagent/open-calm-small",
        tokenizers=tokenizers,
        models_to_merge=models_to_merge,
        initial_lambdas=[0.1, 0.1, 0.1],  # 初期値を設定
        num_epochs=1,
        batch_size=1,
        num_train_samples=2  # テスト用に少数のサンプル
    )

    # 学習前の状態を保存
    initial_lambdas = optimizer.lambdas.clone().detach()
    print("初期状態のλ:", initial_lambdas.cpu().numpy())
    print("初期状態の勾配:", optimizer.lambdas.grad)

    # 1エポック学習
    loss = optimizer.train_one_epoch(batch_size=1)
    
    print("\n学習後:")
    print("損失値:", loss)
    print("λの値:", optimizer.lambdas.detach().cpu().numpy())
    print("λの勾配:", optimizer.lambdas.grad)  # None以外の値になっているはず

    # 勾配の値を確認
    if optimizer.lambdas.grad is None:
        print("警告: 勾配が計算されていません！")
    else:
        print("勾配の統計:")
        print("- 最小値:", optimizer.lambdas.grad.min().item())
        print("- 最大値:", optimizer.lambdas.grad.max().item())
        print("- 平均値:", optimizer.lambdas.grad.mean().item())

    # 最適化ステップ
    optimizer.optimizer.step()
    
    # λの変化量を確認
    delta = optimizer.lambdas.detach() - initial_lambdas
    print("\nλの変化量:", delta.cpu().numpy())
    
    optimizer.optimizer.zero_grad()
    
    print("\n最適化ステップ後:")
    print("λの値:", optimizer.lambdas.detach().cpu().numpy())
    print("λの勾配:", optimizer.lambdas.grad)

    # 5. 各λの値が異なることを確認
    unique_values = len(set(optimizer.lambdas.detach().cpu().numpy()))
    print("\nユニークなλの値の数:", unique_values)
    assert unique_values > 1, "すべてのλが同じ値になっています"

if __name__ == "__main__":
    test_lambda_gradients() 