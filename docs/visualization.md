# BEVFormer 可视化工具使用说明

## 简介

`visualize_results.py` 是一个用于可视化 BEVFormer 模型检测结果的工具。它能够将3D检测结果以多视图方式显示，并在图像上绘制检测到的3D边界框。

## 功能特点

- 支持多视角相机输入的可视化
- 在图像上绘制3D边界框
- 支持自定义得分阈值
- 提供调试模式
- 支持批量处理
- 可自定义输出目录

## 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- Matplotlib
- MMDetection3D
- BEVFormer环境

## 使用方法

### 基本用法

```bash
python tools/visualize_results.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out-dir OUT_DIR] [--score-thr SCORE_THR] [--debug]
```

### 参数说明

- `CONFIG_FILE`: 配置文件路径
- `CHECKPOINT_FILE`: 模型检查点文件路径
- `--out-dir`: 可视化结果输出目录，默认为 'vis_results'
- `--score-thr`: 检测框得分阈值，默认为 0.3
- `--debug`: 启用调试模式，显示更多信息

### 示例

```bash
# 使用默认参数
python tools/visualize_results.py configs/bevformer/bevformer_base.py checkpoints/bevformer_base.pth

# 自定义输出目录和得分阈值
python tools/visualize_results.py configs/bevformer/bevformer_base.py checkpoints/bevformer_base.pth --out-dir my_results --score-thr 0.5

# 启用调试模式
python tools/visualize_results.py configs/bevformer/bevformer_base.py checkpoints/bevformer_base.pth --debug
```

## 输出说明

可视化结果将保存在指定的输出目录中，包括：

1. 原始图像
2. 带有3D边界框标注的图像
3. 多视图组合图像

## 注意事项

1. 确保在正确的conda环境中运行脚本
2. 检查点文件必须与配置文件匹配
3. 输入图像必须符合模型要求
4. 建议使用GPU进行推理以获得更好的性能

## 常见问题

1. **环境问题**
   - 确保已激活正确的conda环境
   - 检查所有依赖包是否正确安装

2. **内存问题**
   - 如果处理大量图像时遇到内存不足，可以调整批处理大小
   - 考虑使用较小的图像分辨率

3. **可视化效果**
   - 如果边界框显示不正确，检查投影矩阵是否正确
   - 调整得分阈值以获得更好的可视化效果

## 代码结构

主要函数说明：

- `parse_args()`: 解析命令行参数
- `project_3d_box_to_2d()`: 将3D边界框投影到2D图像平面
- `draw_3d_box()`: 在图像上绘制3D边界框
- `get_projection_matrix()`: 获取特定视图的投影矩阵
- `denormalize_image()`: 将归一化的图像转换回原始图像
- `visualize_multi_modality_result()`: 多模态数据的可视化函数

## 贡献指南

欢迎提交问题和改进建议：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。 