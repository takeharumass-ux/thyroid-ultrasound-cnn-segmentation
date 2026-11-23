# 甲状腺超声 CNN 实战项目

这是一个可直接在 VS Code 中运行的课程项目：使用轻量 U-Net（CNN）自动分割甲状腺超声图中的结节区域。

## 目录怎么找

- `main.py`：唯一主入口。
- `data/raw/TN3K`：完整数据集，图像和对应 mask。
- `data/processed`：运行 `prepare` 后生成的缓存数据。
- `src`：数据处理、U-Net、训练、评价和预测代码。
- `models`：模型权重；已放入一份已训练好的 C 组权重。
- `results`：训练日志、测试指标和预测图片。
- `run_*.bat`：Windows 快速运行入口。

## 在 VS Code 中运行

建议使用已验证的 Python 环境：`D:\Anaconda\envs\pytorch_learning\python.exe`。

```powershell
python main.py prepare
python main.py evaluate --experiment C
python main.py predict --image data/raw/TN3K/test-image/0000.jpg
```

已有权重时，前两条命令可以直接查看结果；预测图会写入 `results/predictions`。

从头训练 C 组：

```powershell
python main.py train --experiment C --epochs 30
```

快速检查代码和显卡：

```powershell
python main.py train --experiment C --epochs 1 --limit 16
```

三组设置：A 是基础 BCE，B 增加数据增强，C 在 B 上增加 Dice 损失。训练结束后权重写入 `models`，逐轮记录写入 `results`。

## 数据说明

TN3K 已经完整放入 `data/raw/TN3K`，包含 3493 张图像和 3493 张对应 mask。`data/raw/tn3k-trainval-fold0.json` 保存训练/验证划分。运行 `prepare` 后，程序会保持原始图像比例缩放到 192×192，并排除补边区域的损失与评价。生成的 `data/processed` 缓存被忽略，避免把可重建的二进制缓存重复上传。
