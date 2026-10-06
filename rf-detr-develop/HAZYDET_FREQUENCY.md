# HazyDet 目标与背景的空间频率分析

`visualize_hazydet_frequency.py` 比较目标框与同图、同尺寸的未标注背景块，生成频谱图、频段统计和可追溯的 CSV。只需要图像和 COCO 标注，不需要训练模型、权重或 GPU。

## 运行

在项目根目录、Python 3.10 及以上环境中运行：

```bash
python -m pip install numpy matplotlib pillow
python visualize_hazydet_frequency.py --dataset-dir /root/autodl-tmp/HazyDet_RFDETR --split valid --max-images 200 --output-dir output/hazydet_frequency_valid
```

默认匹配现有训练脚本使用的数据结构：

```text
HazyDet_RFDETR/
  valid/
    _annotations.coco.json
    image001.jpg
    ...
```

图像相对路径由 COCO `images[].file_name` 决定，允许子目录。Windows 下将数据路径替换为自己的路径，例如 `--dataset-dir "D:/datasets/HazyDet_RFDETR"`。

如果图像和标注不在这个结构中，显式指定两者：

```bash
python visualize_hazydet_frequency.py --annotations /data/HazyDet/annotations/instances_val.json --image-dir /data/HazyDet/images/val --output-dir output/hazydet_frequency_custom
```

这些路径只是示例，应替换为实际位置。此脚本读取 COCO 的 `images`、`annotations`、`categories`，不直接读取 VOC XML 或 YOLO TXT。

分析全部图像和全部符合尺寸条件的框：

```bash
python visualize_hazydet_frequency.py --dataset-dir /root/autodl-tmp/HazyDet_RFDETR --split valid --max-images 0 --max-pairs-per-image 0 --output-dir output/hazydet_frequency_all
```

## 输出与读图

| 文件 | 内容与读法 |
| --- | --- |
| `summary.png` / `.pdf` | 四个汇总面板，见下文 |
| `example_001.png` / `.pdf` 等 | 原图上的全部框和选中区域、目标/背景原始裁剪、共享色阶的二维频谱、径向能量分布 |
| `pairs.csv` | 每个配对的图像、类别、标注 ID、原图坐标、目标面积、绝对能量与各频段占比 |
| `per_image.csv` | 每张图内的配对平均值；汇总图按图像等权计算 |
| `radial_spectra.csv` | 每张图的目标和背景径向能量分布，可重新作图 |
| `run_summary.json` | 参数、随机种子、标注 SHA256、实际样本数、跳过原因、统计结果和方法限制 |

汇总图左上是径向频谱，越靠右变化越快；阴影是图像间 ±1 标准差，不是置信区间。纵轴是每个频率环带占总交流能量的比例，不是环带内每个频率点的平均功率密度。

右上比较低、中、高频占比。默认分界为 0.08 和 0.20 cycles/pixel，分别对应约 12.5 和 5 像素的空间周期。分界是探索性设置，不是 HazyDet 的既定标准。某组高频占比较高，说明其纹理能量相对更集中在快速变化部分，并不意味着绝对高频能量一定更强。

左下比较绝对交流能量，每个点是一张图：位于对角线上方，表示该图中采样目标框的平均交流能量大于配对背景。交流能量去除了平均亮度，反映局部亮度变化强弱，不是平均亮度本身。

右下检查目标大小的影响，每个点是一对区域。纵轴大于零表示该目标框的高频占比高于对应背景。不要把一张图中多个框当作独立图像样本用于显著性检验。

终端输出 `Image-weighted high-frequency fraction difference (object - background)`：正数表示本次有效样本中目标的高频占比平均更高，负数表示背景更高。单个平均数不能证明差异稳定或具有统计显著性。

## 计算口径

1. 按固定随机种子选择图像，每张图默认最多随机尝试 10 个目标框，保留原始像素尺寸，不缩放。
2. 为每个目标框在同一张图中均匀随机提出同宽、同高的背景矩形，接受首个不接触任何标注框及其边距的矩形。crowd、ignore 和过小目标也会被排除在背景外，但不作为目标样本。
3. 将 RGB 转为 `[0,1]` 灰度 `0.299R + 0.587G + 0.114B`，减去 Hann 加权均值，乘二维 Hann 窗，再做正交归一化 FFT。窗口减轻裁剪边界造成的谱泄漏；它也会降低框边缘的权重。
4. 功率为 `abs(FFT)**2 / sum(window**2)`，去除零频分量。功率总和为窗口校正后的交流能量；按径向频率求和并除以总能量，得到频段占比。常量区域的占比无意义，因此整对跳过并记录。
5. 各配对先在每张图内求平均，再对有效图像等权求平均。径向频率为 `sqrt(fx**2 + fy**2)`，单位是原图 cycles/pixel。二维角落可达到约 0.707，单轴 Nyquist 频率是 0.5。

FFT 功率与频率轴的定义可参考 [NumPy FFT 文档](https://numpy.org/doc/2.2/reference/routines.fft.html)；窗口化功率谱估计可参考 [SciPy periodogram 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)。此脚本使用 NumPy 实现，不需要安装 SciPy。

## 可调整参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--max-images` | 200 | 随机选择图像数，0 表示全部 |
| `--max-pairs-per-image` | 10 | 每图最多尝试多少个目标框，0 表示全部；成功配对数可能更少 |
| `--min-size` | 16 | 裁剪后目标框最短边至少多少原始像素 |
| `--bg-margin` | 4 | 背景与所有标注框之间的排除边距 |
| `--bg-tries` | 500 | 每个目标尝试多少个背景位置；失败不等于不存在可用背景 |
| `--low-cut` / `--high-cut` | 0.08 / 0.20 | 低/中频及中/高频分界，cycles/pixel |
| `--bins` | 32 | 径向频谱的等宽分箱数 |
| `--examples` | 6 | 最多输出几张示例，每张图只画一对；设为 0 可加快运行 |
| `--seed` | 42 | 复现抽样的随机种子 |

可改变种子、分界和最小框尺寸检查结论是否稳定，例如 `--seed 123 --low-cut 0.06 --high-cut 0.15`。不同设置请用不同输出目录，避免旧示例图残留造成混淆。

## 结论边界

不能预设“目标就是高频，背景就是低频”。目标也有低频结构，背景也可能有边缘、树叶、建筑纹理等高频内容。频率描述图像变化的尺度，不直接标识目标语义。

目标框中仍然包含背景；框外仅表示未标注区域，不能保证没有漏标目标。同图、同尺寸配对控制了部分图像和尺寸差异，但没有匹配深度、局部雾浓度、光照或背景材质。多个背景块可以重叠，密集场景和大框更难配对。因此结果只适用于成功采样的区域，必须同时检查 `run_summary.json` 的覆盖率和跳过计数。

小目标的频率分辨率较粗，默认短边小于 16 像素的框会跳过；可以降低阈值，但无法通过增加频谱分箱补回原本不存在的分辨率。类别保存在 `pairs.csv`，必要时应分车型、目标大小等进一步分析。

这些图是描述性探索，不直接验证频率模块能否改善检测，也不能单凭有雾数据证明“雾导致了某种频率变化”。后者需要额外的对照设计。

## 验证

```bash
python tests/visualize/test_hazydet_frequency.py
```

测试使用合成图像检查已知频率、能量缩放、背景排除和完整 COCO 运行。合成测试结果不能当作 HazyDet 的实验结果。
