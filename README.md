# 二维 EM 神经元实例分割

深度学习导论双人课程项目。项目用 U-Net 预测邻接像素亲和度，再用分水岭生成神经元实例。

- [课程任务书](Project%20copy.pdf) · [中文翻译](Project%20copy_任务书_中文翻译.md)
- [协作和模型更新约定](AGENTS.md)
- [原基线与失败案例](analysis/GPT-001/README.md) · [v1](analysis/GPT-002/README.md) · [v2](analysis/GPT-003/README.md) · [v3](analysis/GPT-004/README.md)

## 运行

使用 Python 3.12 和实验配置记录的依赖版本。训练与评估使用 PyTorch、NumPy、SciPy 和 Pillow，绘图还需要 Matplotlib。
从共享数据副本恢复 `data/{train,val,test}/{images,labels}/`，对照 [数据清单](data/manifest.json) 核对文件。历史分析中的权重和逐图预测链接对应共享的 `runs/` 文件，复用时按原相对路径恢复。以下命令从项目根目录运行，使用新的输出路径。

```powershell
python baseline/train.py --out runs/baseline_repro
python baseline/evaluate.py --checkpoint runs/baseline_repro/best.pt --split val --out runs/baseline_repro/val
```

历史指标、配置和训练日志保存在 `runs/`，各轮分析和复现细节见上面的版本记录。根目录的 Git 标签 `project-import` 指向首次导入现有成果的快照。
