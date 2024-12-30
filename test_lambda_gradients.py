import torch
from optimize_lambdas_cross_entropy import LambdaOptimizerCrossEntropy

def test_lambda_gradients(optimizer: LambdaOptimizerCrossEntropy):
    """
    λの勾配計算が正しく行われているかをテストする関数
    """
    # 初期状態の確認
    print("初期λ:", optimizer.lambdas.data)
    print("requires_grad:", optimizer.lambdas.requires_grad)

    # 1回の損失計算 (GSM8Kの1バッチで試す - バッチサイズ2)
    batch_data = optimizer.gsm8k_data[:2]
    loss = optimizer.compute_gsm8k_batch_loss(batch_data, optimizer.tokenizers[0])

    # 勾配計算前の状態を確認
    print("\n勾配計算前:")
    print("λの勾配:", optimizer.lambdas.grad)

    # 勾配計算
    loss.backward()

    # 勾配計算後の状態を確認
    print("\n勾配計算後:")
    print("λの勾配:", optimizer.lambdas.grad)
    print("損失値:", loss.item())

    # 勾配の値がNoneでも0でもないことを確認
    assert optimizer.lambdas.grad is not None, "勾配がNoneです"
    assert not torch.all(optimizer.lambdas.grad.eq(0)), "すべての勾配が0です" 