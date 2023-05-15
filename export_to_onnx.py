import torch

# PyTorchモデルの読み込み
model = torch.load('best.pt')  # モデルファイル名を指定

# モデルを評価モードに設定
model.eval()

# ダミーの入力テンソルを作成
dummy_input = torch.randn(1, 3, 64, 64)  # 入力テンソルの形状に合わせて適切な値を指定

# ONNX形式に変換
onnx_file_path = 'best.onnx'  # 保存するONNXファイルのパスを指定
torch.onnx.export(model, dummy_input, onnx_file_path)
