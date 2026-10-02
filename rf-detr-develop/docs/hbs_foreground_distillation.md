# HBS + 清晰图前景蒸馏

本实现是训练时的特征蒸馏。雾图学生的 backbone + projector 输出（HBS 之前）
与清晰图教师对应层的目标框区域进行 ROIAlign；每个目标采样为 3×3，沿通道做 L2
归一化，再计算半平方距离，对目标和特征层平均。小目标不会因为面积小而被大目标淹没。
框内可能包含背景，因此这是框级前景监督，不是像素级前景分割。

教师只加载同结构检测器的 backbone + projector，严格校验所有参数，不运行教师检测头。
教师冻结、eval、no_grad；学生同时学习雾图检测、HBS 辅助检测和前景蒸馏。
推理与导出不使用教师、配对图或蒸馏分支。训练增加教师前向的显存和时间；Lightning
训练断点包含教师状态，现有 EMA 也会复制 Lightning 模块，需预留额外训练显存。
原有学生 `.pth` 导出不包含教师。

## 训练

### 先训练清晰图教师

教师**不需要 HBS**。蒸馏加载器只读取 backbone + projector，忽略检测头与 HBS 权重。
推荐使用新增脚本训练普通 RF-DETR Small，保持学生默认的 512 分辨率和模型配置：

```bash
python train_clear_teacher.py --prepare-only
python train_clear_teacher.py
```

脚本使用 `/root/autodl-tmp/HazyDet/train/images` 中的清晰图训练，并沿用
`/root/autodl-tmp/HazyDet_RFDETR/train/_annotations.coco.json` 的训练划分和框坐标。
它在独立目录 `HazyDet_RFDETR_clear_teacher` 生成引用原始图片绝对路径的 COCO 索引，
不复制图片，不修改原始数据。这些索引应在训练服务器生成，迁移服务器时需重新生成。

默认验证集为原数据集的 `valid`，通常是雾图：训练仍只使用清晰图，但最佳权重按照
该雾天验证集的效果选择。若要用清晰图验证，提供 `--clear-valid-dir /清晰验证图目录`。
不要将训练图用作验证图。原图若与 COCO 文件名不一致，传 `--pair-manifest pairs.json`；
映射格式与学生一致，只使用其中的 `clear` 字段。清晰验证图可另传 `--valid-manifest`。
若清晰原图与导出标注尺寸不一致，脚本会报错，需先准备匹配原图坐标的标注。

训练完成后启动学生：

```bash
python train_hazydet.py \
  --teacher-weights /root/autodl-tmp/rf-detr/output/hazydet_clear_teacher/checkpoint_best_total.pth
```

教师训练参数默认与现有学生入口一致：36 epochs、batch size 4、梯度累积 4、lr=1e-3、
EMA 开启；可使用 `--epochs`、`--batch-size`、`--grad-accum-steps`、`--lr` 等参数调整。
这只是对照实验起点，不代表已验证的最佳教师训练配置。

### 已有教师权重时

先使用**同结构、同分辨率** RF-DETR Small 在清晰训练集上训练教师。
不要把雾天 baseline 权重当作清晰图教师，也不要使用验证/测试集训练教师。
本地 Windows 无法直接访问下面的 Linux 数据目录，请在原训练服务器运行：

```bash
python train_hazydet.py \
  --teacher-weights /你的清晰图教师/checkpoint_best_total.pth
```

脚本默认值：

| 参数 | 默认值 |
| --- | --- |
| `--dataset-dir` | `/root/autodl-tmp/HazyDet_RFDETR`（沿用 COCO 标注与验证集） |
| `--clear-dir` | `/root/autodl-tmp/HazyDet/train/images` |
| `--hazy-dir` | `/root/autodl-tmp/HazyDet/train/hazy_images` |
| `--fg-distill-coef` | `0.1`（实验起点，尚未证实最优） |
| HBS 系数 | `0.25` |
| 蒸馏 warmup | 前 3 个 epoch 的有效系数为 0.1/3、0.2/3、0.1 |
| 输出目录 | `/root/autodl-tmp/rf-detr/output/hazydet_small_hbs_fgkd` |

实际总损失为原检测损失 + 0.25 × HBS 辅助检测损失 + 有效蒸馏系数 × 前景损失。
监控 `train/loss_fg_distill` 和 `train/loss_fg_distill_weighted`，后者才是对总损失的贡献。
`fg_distill_coef=0` 关闭整个教师/配对路径。HBS-only 对照可用
`--fg-distill-coef 0 --output-dir /独立输出目录`，无需教师参数。

## 图像配对

默认要求 COCO `images[].file_name` 在清晰图和雾图目录下都能直接找到。
保留文件名及相对子目录，不猜测后缀，不自动去除 Roboflow 重命名部分。
如果导出文件名与原图不同，提供 `--pair-manifest /路径/pairs.json`：

```json
{
  "COCO标注中的文件名.jpg": {
    "clear": "清晰目录下的原图.jpg",
    "hazy": "雾图目录下的合成雾图.jpg"
  }
}
```

每条训练记录都必须有映射。启动时检查所有路径；读取时检查清晰图、雾图及 COCO
标注尺寸。若 Roboflow 已缩放/裁剪原图，仅修改名字不够，需要先准备对应坐标的标注。
清晰图与雾图还须内容一一对应、像素配准，尺寸相同本身不能证明配对正确。

配对图共用 CPU Albumentations 的裁剪、缩放、翻转等几何参数，共用批量多尺度缩放。
单独的亮度、模糊等像素变换只作用于学生。若自定义 `OneOf/Sequential` 将几何与
像素变换混在同一个容器，则容器整体同步到教师；建议将像素变换放在独立项。
本版仅支持 COCO/Roboflow 目标检测及 `augmentation_backend="cpu"`。

## 验证

完整项目测试环境中运行：

```bash
python -m pytest tests/test_foreground_distillation.py tests/test_foreground_distillation_training.py
```

测试涵盖配对翻转/裁剪/缩放、归一化、空目标、梯度隔离、严格教师加载、文件映射与
COCO 数据流、多尺度训练钩子、损失加权和蒸馏开关。真实检测精度需在相同划分、相同训练预算下对比 baseline、HBS、HBS+KD；
重点检查小目标 AP、召回率和误检。不要仅凭蒸馏损失下降判断检测效果。
