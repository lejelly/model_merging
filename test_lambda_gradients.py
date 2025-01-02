import torch
from optimize_lambdas_cross_entropy import LambdaOptimizerCrossEntropy, MergedModelWrapper
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
        num_epochs=2,
        batch_size=1,
        num_train_samples=2  # テスト用に少数のサンプル
    )

    print("\n=== 基本的な勾配テスト ===")
    # 学習前の状態を保存
    initial_lambdas = optimizer.lambdas.clone().detach()
    print("初期状態のλ:", initial_lambdas.cpu().numpy())
    print("初期状態の勾配:", optimizer.lambdas.grad)
    print(f"requires_grad: {optimizer.lambdas.requires_grad}")

    # 1エポック学習
    loss = optimizer.train_one_epoch(batch_size=1)
    
    print("\n学習後:")
    print("損失値:", loss)
    print("λの値:", optimizer.lambdas.detach().cpu().numpy())
    print("λの勾配:", optimizer.lambdas.grad)

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

    print("\n=== 詳細な勾配テスト ===")
    # MergedModelWrapperの作成
    merged_model = MergedModelWrapper(
        pretrained_model=optimizer.pretrained_model,
        task_vectors=optimizer.task_vectors,
        lambdas=optimizer.lambdas
    )
    merged_model.to(optimizer.device)

    # 小さなバッチで順伝播と逆伝播を実行
    batch_size = 1
    test_data = optimizer.gsm8k_data[:batch_size]
    
    # 勾配をクリア
    optimizer.optimizer.zero_grad()

    # 損失計算
    detailed_loss = optimizer.compute_gsm8k_batch_loss(
        test_data,
        optimizer.tokenizers[0],
        merged_model
    )

    print(f"\n詳細テストの損失値: {detailed_loss.item()}")
    print(f"損失のrequires_grad: {detailed_loss.requires_grad}")

    # 逆伝播
    detailed_loss.backward()

    # 勾配の確認
    print("\n逆伝播後の勾配:")
    if optimizer.lambdas.grad is None:
        print("警告: λ.grad is None!")
    else:
        print(f"λの勾配: {optimizer.lambdas.grad.cpu().numpy()}")
        
        # 勾配が0でないか確認
        if torch.all(optimizer.lambdas.grad == 0):
            print("警告: すべての勾配が0です！")
        else:
            print("✓ 非ゼロの勾配が存在します")

    # 最適化ステップ実行
    old_lambdas = optimizer.lambdas.clone().detach()
    optimizer.optimizer.step()
    
    # λの値が更新されたか確認
    print("\n最適化ステップ後のλの値:")
    print(f"更新前: {old_lambdas.cpu().numpy()}")
    print(f"更新後: {optimizer.lambdas.detach().cpu().numpy()}")
    
    if torch.allclose(old_lambdas, optimizer.lambdas):
        print("警告: λの値が更新されていません！")
    else:
        print("✓ λの値が正しく更新されました")

    return not torch.allclose(old_lambdas, optimizer.lambdas)

if __name__ == "__main__":
    success = test_lambda_gradients()
    print(f"\n勾配テスト: {'成功' if success else '失敗'}!")