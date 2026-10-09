# Projector 清晰 / 雾天诊断

## 随机抽取 200 张（默认模式）

已按当前实验配置设置默认数据目录 `/root/autodl-tmp/HazyDet_RFDETR`、split `valid`，
权重路径 `/root/autodl-tmp/rf-ddetr/output/hazydet_small_baseline/checkpoint_best_total.pth`。
注意权重路径中的 `rf-ddetr` 按用户提供的拼写保留；如果服务器实际目录是 `rf-detr`，请通过参数修改。

在训练服务器项目根目录运行：

```bash
python visualize_projector.py --output-dir output/projector_random200 --device cuda
```

完整写法：

```bash
python visualize_projector.py \
  --dataset-dir /root/autodl-tmp/HazyDet_RFDETR --split valid \
  --checkpoint /root/autodl-tmp/rf-ddetr/output/hazydet_small_baseline/checkpoint_best_total.pth \
  --num-samples 200 --seed 42 \
  --output-dir output/projector_random200 --device cuda
```

递归查找该 split 的图片，无放回抽样；不足 200 张会报错。固定 seed 和相同文件列表得到相同样本。
`samples.json` 保存实际样本路径、天气分组和内容哈希。输出目录必须为空且位于输入目录之外。
只抽样一次，逐张推理，再利用保存的能量图统一各层的全样本色标，不需要第二次推理。

若清晰/雾天分开存放，可用真实目录标签进行组间比较（200 张总计，每组 100 张）：

```bash
python visualize_projector.py --clear-dir /data/clear --haze-dir /data/haze \
  --num-samples 200 --seed 42 --output-dir output/projector_weather200 --device cuda
```

任意单一图片目录也可通过 `--image-dir /data/images` 指定。
混合目录不自动推断天气标签，也不会把“较清晰”冒充无雾；缺少分组时只分析总体分布及图像对比度的相关性。

输出包括：

- `images/0000.png` 等：每张原图和全部 pre/post 热图，编号对应抽样清单顺序。
- `maps/0000.npz` 等：所有样本的原生分辨率 RMS 图。
- `statistics.csv`：逐图逐层的 RMS 均值、空间变异系数、图像对比度和形状。
- `summary.csv`、`summary.png`：分层分组统计和分布图。
- `analysis.md`：由本次实测统计生成的中文报告，包括组间差异（有天气标签时）、局限和下一步验证。
- `run.json`、`samples.json`：配置、统一色标和抽样记录。

需要完整特征再加 `--save-raw`（200 张的所有 CHW 特征可能占用较多磁盘空间）。
报告是描述性诊断，不计算 AP/召回、不执行目标匹配，不会自动断言 Projector 是性能瓶颈。

## 显式配对模式

在训练服务器的项目根目录、已安装本仓库的 Python 环境运行；无需训练：

```bash
python visualize_projector.py \
  --checkpoint /root/autodl-tmp/rf-detr/output/hazydet_small_hbs/checkpoint_best_total.pth \
  --clear /path/to/clear.jpg \
  --haze /path/to/haze.jpg \
  --output-dir output/projector_diagnostics \
  --device cuda --save-raw
```

替换图片路径。多个样本可写 `--clear c1.jpg c2.jpg --haze h1.jpg h2.jpg`，按顺序对比。
只有同一场景、同一视角、逐像素配准的清晰/雾天图片才添加 `--aligned`；相同尺寸不代表配准。
没有真实配对数据时可先作非配对探索，但不能把场景差异归因于雾。

每对图片保存到 `pair_000/` 等目录：

- `comparison.png`：原图、所有 Projector 输入层（pre）和输出尺度（post）的 RMS 特征叠加图。
- `activation_maps.npz`：原生分辨率能量图；配准模式另外保存逐位置差异图。
- `statistics.csv`：各层能量均值、空间标准差、统一色标；配准模式增加余弦相似度、有效比例和相对 L2。
- `clear_features.npz`、`haze_features.npz`：`--save-raw` 时保存完整 CHW 特征。
- 总目录的 `run.json`：权重路径、图像路径、模型分辨率、输入层索引和输出尺度。

同层清晰/雾天共享色标；不同 pre/post 层的通道空间和尺度不同，不能直接相减，也不能凭颜色强弱比较信息保留量。
热图是通道 RMS 激活能量，并非注意力、目标概率或检测准确率。
输出层可能融合多个输入层，所以 pre_0 与 post_0 并非一一对应。

本仓库 `LWDETR.forward` 仅在 `self.training and self.hbs is not None and targets is not None` 时运行 HBS。
脚本通过 `from_checkpoint(..., hbs_enabled=False)` 加载并使用 eval 模式，不运行 HBS，不修改原权重文件。
HBS 权重中多出的 `hbs.*` 参数可能被加载器报告为 unexpected keys；除此之外的缺失、额外参数或形状问题必须检查。
不要用随机初始化模型代替无法正确加载的权重。

已经训练完的模型可以直接分析。HBS 训练过的权重保留了辅助监督对主干的影响，关闭分支不会消除训练历史。
若要判断原始 RF-DETR 的缺陷，应优先使用之前保存的无 HBS 权重；没有这样的权重才需要独立训练无 HBS 基线。
`train_hazydet.py` 已将 HBS 关闭，并将新训练输出目录改为 `hazydet_small_baseline`。

## 如何定位待验证的问题

1. 同场景雾天在 Projector 输入处已经出现明显差异：优先检查输入缩放、目标像素大小和编码器表征。
2. 输入表征相对稳定，但输出的天气差异增大、目标区域与背景难以区分：将 Projector 融合/投影列为候选瓶颈。
   不同层特征空间不等价，余弦/L2 的跨层变化只能作为线索，不能单独证明信息损失。
3. 特征仍保留目标区域结构，但检测漏检：进一步检查 decoder 查询、分类置信度、定位误差和阈值。

用真实标注定义目标/背景区域，结合检测结果逐个查看漏检的小目标；最终在固定验证集上按天气与目标尺寸统计 AP/召回，
再进行一次只改一个因素的消融。单对图片或激活强度不能证明 RF-DETR 在恶劣天气下的结构性缺陷。

本地组件验证：

```bash
python -m unittest discover -s tests/visualize -p test_projector_diagnostics.py
```
