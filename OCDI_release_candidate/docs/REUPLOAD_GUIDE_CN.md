# OCDI 重新发布与 ACS 数据流程详细教程

适用范围：针对公开仓库提交 50874310cbb2dc876df49d086b63742af5c06a3a，以及作者提供的 MB3R1.PY，整理的 MB3R1 修复候选包。这里没有正式实验数据、训练权重、各基线源代码、Stage-U 或 MB4 代码。因此它能修复已知打包与数据接口问题，不能直接被称为“论文所有设置完整复现、保证不会再被质疑的最终版”。

## 1. 先明确发布方式

在原仓库删除文件并重新上传，会产生删除和新增的提交，历史仍然存在。删除整个仓库再同名建立，也不能保证旧克隆、引用和历史副本消失。修复不需要消除历史。

建议：保留 https://github.com/yqx7150/OCDI，在分支中完成一次可核查的修复，验证后合并，再建立版本发布。审稿人的意见是 packaging issue，公开修复记录正好能够证明已处理。

下文给出“本地替换当前全部受跟踪文件，再一次提交”的方法，满足重新整理整个文件树的需要。不会先让公开 main 变成空仓库，也不改写历史。

不要执行 force push、删除远端仓库或创建无历史分支来隐藏本次修复。

## 2. 本次代码检查发现了什么

| 位置 | 观察到的事实 | 处理 |
| --- | --- | --- |
| test/test.py | 导入缺失的 ocdi.data.packed | 补齐 packed.py |
| train/train.py | 导入缺失的 ocdi.data.mb3 | 从完整训练源码恢复数据集 |
| .gitignore | 未限定位置的 data/ 会忽略 ocdi/data/ | 改为 /data/ 与 /datasets/ |
| pyproject.toml | 只打包 ocdi*，未包含文档入口 | 包含 train/test/scripts |
| 原训练数据集 | 使用完整 K1 和 K2+K3 生成 x_t，计算监督损失 | 这是监督训练；不能据此认定测试泄露 |
| 原训练/验证 | 仅根据目录读取样本，没有患者划分证据 | 补真实 subject/volume manifest |
| 原 best_model.pt 保存逻辑 | 没有保存 args，逐 epoch 权重却有 args | 新 best checkpoint 保存 args；旧权重需找回真实配置 |
| 原默认参数 | 训练 coils=12，公开测试默认 coils=16 | 必须按正式权重对齐，不能直接使用两边默认值 |
| 原测试加载 | 一个 packed 文件同时提供输入与完整标签 | 候选版将测量、ACS、评分标签分开 |
| 公开参考拟合 | fit_slf_reference 的拟合 patch 限定在中心频带 | 这是有利证据；进一步强制参考文件外部为零 |
| 公开测试说明 | 将参考称作 independently acquired | 按作者声明改成 retrospectively simulated ACS |
| 评分函数 | 默认用 GT 做强度基准、gain matching 和 clipping | 属于评分后处理，但必须统一披露，不能隐藏或跨方法混用 |

原文件没有 slice_ref/ppa_ref 的生成代码，也没有 SG/SPSG/SENSE/RAKI/KDM/ROGER/FM 的实现，所以不能用这份训练文件证明所有方法的历史校准流程。

## 3. 数据预算应该怎样定义

设第 s 层的单层多线圈 k-space 为 k_s，已知 CAIPI 调制为 D_s，多带采样掩膜为 P_R，中心 ACS 掩膜为 A：

- 回顾性多带输入：y = P_R × sum_s(D_s × k_s)。
- 单层校准信息：c_s = A × k_s。
- 方法需要 SMS 坐标系时，可用已知调制 D_s × c_s；不能再额外使用 ACS 外的单层数据。
- 完整 k_s 另外保留，供离线模拟和最终评分。

务必区分：

1. 裁剪每一层 single-band k-space 的 ACS，提供 slice-resolved reference。
2. 只裁剪叠加后的 multiband k-space。

二者信息量不同，第二种一般不能直接提供第一种逐层参考。你描述的是第一种：利用单层原始数据的中心 ACS，回顾性模拟逐层参考的可用性。应如实称为“回顾性模拟的单层校准参考”，不能称“独立采集参考”，也不能说只靠叠加输入就获得了这些逐层数据。

校准来自当前测试对象的允许 ACS，并不自动构成训练/测试泄露；问题在于有没有超出声明的数据预算，或者使用完整测试标签训练、选权重或调参。这需要流程和实现证据。

## 4. 先划分受试者，再生成所有样本

从真实原始来源建立 manifest，列为：

sample_id, subject_id, source_volume, slice_indices, split, path

- subject_id 使用可追溯的匿名标识，不编造、不靠排序猜测。
- 同一受试者的不同切片、组合、噪声版本、扫描/对比度，原则上属于同一 split。
- source_volume 标识原始体数据；slice_indices 明确组内各层来自哪些切片。
- 先划分 train/val/test，再分组、增强和加噪。
- 如果官方公开集合被你进一步划分，说明“本文内部划分”，不能冒称官方测试集。
- 所有全局模型、超参数和 checkpoint 只用训练/验证集合选择。
- 测试时允许的 scan-specific ACS 校准，与跨受试者训练的模型学习应分开叙述。

包内 docs/splits_template.csv 只有表头，需要填真实记录。检查命令：

```powershell
python -m scripts.audit_splits --manifest "E:\OCDI_data\splits.csv" --verify_files
```

path 相对 manifest 所在目录解析，也可使用绝对路径。工具检查声明的 subject/volume 跨集合重复，以及可选的文件路径和字节完全相同；它不能证明匿名 ID 与真实患者之间的映射正确。

## 5. 补齐数据模块的具体内容

候选包包括：

- ocdi/data/__init__.py
- ocdi/data/packed.py
- ocdi/data/mb3.py

mb3.py 恢复训练数据集的主要行为：

- K23 = K2 + K3；保留原有可选 /sqrt(2)。
- t 与 alpha(t) 的取法按原源码。
- x_t = K1 + alpha(t) × K23。
- 保留 KMB 来源、归一化、裁剪/填充及训练期线圈处理。
- 训练标签可以用于监督损失。
- 用明确 HWC 布局替代基于最小维度猜测轴。

候选版与原源码的合成 HWC 数据比较见 VALIDATION.md。真实旧文件仍需逐个确认其实际维度；不能把合成测试等同于临床数据兼容验证。

单独的 load_packed_mb3_mat 可服务训练/离线模拟：如果存在 KMB，保留原值；没有 KMB 时才将已经调制到相同 SMS 坐标系的切片求和。它不重复加 CAIPI 相位。

测试使用另外的 load_measurement_mat，只读取 KMB 和 mask。缺少 KMB 不允许从评分标签补输入，输入文件含 K1/K2/K3 时直接报错。

## 6. 测试数据应分成四类

| 目录 | 内容 | 使用者 |
| --- | --- | --- |
| measurements | KMB、实际 mask | 重建入口 |
| slice_ref | ACS 内的逐层 K1/K2/K3，外部为零，另附 acs_mask | 分层核/参考算子等校准 |
| ppa_ref | 同一允许单层 ACS 样本支持 | 按具体算法转换为面内校准输入 |
| ground_truth | 完整 K1/K2/K3 | 保存预测之后的评分 |

目录分开只是一个接口约束，不能证明“独立采集”。真正限制信息的是 ACS 掩膜、禁止越界、清楚的依赖关系及不变性测试。

ppa_ref 和 slice_ref 可以来自同一单层中心 ACS 信息，但不意味着每个方法接收相同的数组语义。SENSE 的 sensitivity 估计可能需要已知相位解调；SG 的参考可能需要 SMS 调制坐标；GRAPPA/RAKI 则应按各实现构造校准源与目标。候选导出器提供同一允许支持，不代替这些算法适配器。

## 7. 离线生成标准输入与参考

使用原来已经完成 CAIPI 处理、维度为 [H,W,C] 的 packed 样本。候选工具不会生成原始 fastMRI 的切片分组，也不会自动确认是否已经施加相位。

MB3R1 示例：

```powershell
python -m scripts.prepare_retrospective --packed_dir "E:\OCDI_data\test_packed" --out_dir "E:\OCDI_data\test_prepared" --coils 16 --acs_lines 32 --R 1 --confirm_sms_frame
```

参数中的 16 和 32 应与正式实验核对；示例不是从训练文件推断出的历史运行配置。

工具会：

1. 读取已调制的 K1/K2/K3 和可选 KMB。
2. 根据 R、offset 创建多带采样 mask。
3. 将测量外缺失位置置零。
4. 每层只保留中心 ACS，外部严格置零。
5. 将完整评分标签另存。
6. 保存形状、ACS 索引、源文件 SHA256、R 与 mask 信息。
7. 拒绝覆盖已有同名输出，避免混合不同版本。

本接口固定 PE 为 W，ACS 零基索引：
start = W//2 - L//2，stop = start + L，截取 [start, stop)。

若原来的 PE 沿另一轴，先按原预处理规则明确转置，不能靠猜。

R>1 的示例生成器支持 --R 2 等设置，但 **候选测试入口只验证 MB3R1**，缺失采样线时会拒绝运行；要复现加速论文结果，需要真正的 Stage-U/面内补全代码。

默认没有将中心 ACS 加入多带采样 mask。--include_mb_acs 会显式加入并改变有效采样率。不要为了方便而修改原实验的 mask。额外逐层参考的扫描开销应另外披露；MB×R 不等于计入校准开销后的净加速。

## 8. 对所有基线采用统一信息预算

更准确的目标是“所有方法获得相同测量、相同允许校准信息；有需要的组件按需使用”，不是强迫每个方法都调用 ppa_ref 和 slice_ref。

| 方法 | 逐项核查 |
| --- | --- |
| SG | 核拟合的 source、target 和整个 patch 均在 ACS 内；后续面内校准也检查 |
| SPSG | 分层项与泄漏抑制项均不能借用 ACS 外单层数据 |
| SENSE | 先限定 ACS，再估计 coil maps；不能先用完整 GT 估图再裁参考 |
| RAKI | 明确 scan-specific 训练仅使用允许 ACS；不能用完整测试切片拟合 |
| KDM | 核查参考、算子、初始值、线圈图及测试适配的全部依赖 |
| ROGER | 区分离线网络训练与测试校准，核查参考、线圈图与初始值 |
| FM | 全局训练使用训练集合；测试条件、初始化、归一化只用允许信息 |
| OCDI | 同样核查 ACS 算子、输入归一化、初始化、Stage-U 和一致性操作 |

KDM/ROGER/FM 的具体组件取决于你实际使用的实现，不能只凭方法名断言它们都需要相同参考。包内 baseline_audit_template.csv 可逐个填写代码版本、路径、ACS、checkpoint hash 和检查结果。

必须继续检查可能绕开参考文件的路径：完整 GT 估 coil maps、全图 PCA/coil compression、白化矩阵、强度尺度、前景 mask、停止条件、超参数搜索和挑选最好 checkpoint。训练阶段的完整标签监督与这些测试路径应区分。

如果检查发现历史实现确实使用了更多单层数据，应披露并按公平协议重跑受影响比较。不能只改 README 或文件名就宣称原结果遵循了中心 ACS。

## 9. 应保存的两类不变性检查

第一类：评分标签隔离。

固定 KMB、mask、ACS 与权重，更换 ground_truth 或完全不提供它。保存的预测必须相同；只有分数可以改变。

第二类：校准支持隔离。

固定 ACS 内数据，扰动原单层数据 ACS 外部分，重新拟合参考算子。系数必须相同。这里不能重新生成测量 y，否则 y 随真实输入改变，预测改变并不代表泄露。

包内 synthetic smoke 检查第一类，数据测试检查第二类；各基线与真实数据仍需补同样证据。

评分函数的 gain matching 是在预测后使用 GT 的评分变换，不能称“重建完全无需 GT，所以评分没有任何 GT 后处理”。应明确披露并对所有方法采用相同设置。若更换指标定义，应重算全部对应方法，而不是只改变 OCDI 的分数。

## 10. 全新安装与合成验收

先把候选包解压到一个独立目录，暂时不要修改 GitHub。

Windows PowerShell：

```powershell
Set-Location "C:\OCDI_release_candidate"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-cpu-tested.txt .
python -m pip check
python -m train.train --help
python -m test.test --help
python -m unittest discover -s tests -v
python -m scripts.check_original_parity --original "reference\MB3R1_original.py"
python -m scripts.smoke_test
```

如果 PowerShell 执行策略阻止激活，可直接调用 .venv\Scripts\python.exe 执行上述命令，无需改全局策略。

然后离开源码目录，确认普通安装有效：

```powershell
Set-Location "C:\"
C:\OCDI_release_candidate\.venv\Scripts\python.exe -c "import ocdi; print(ocdi.__file__)"
C:\OCDI_release_candidate\.venv\Scripts\python.exe -m train.train --help
C:\OCDI_release_candidate\.venv\Scripts\python.exe -m test.test --help
C:\OCDI_release_candidate\.venv\Scripts\python.exe -m scripts.smoke_test
```

打印出的 ocdi 路径应在该虚拟环境的 site-packages 中。使用新环境，不依赖旧机器上的 PYTHONPATH、另一个本地 ocdi 副本或 editable 安装。

这套 CPU 合成配置仅检查打包与基本执行。它不复现论文 PSNR/SSIM，不替代 GPU/真实数据验证。

## 11. 正式权重与真实样本验收

先找回生成正式权重的实际参数，而不是猜测默认参数：

- coils、base_ch、T、alpha schedule。
- xattn 与 auxiliary head 结构。
- amp_norm、KMB 来源、K23 可选缩放。
- 32 ACS、kernel、正则化与指导参数是否为最终设置。
- 真实权重 SHA256、源训练提交和 subject split。
- 原始 MATLAB 轴顺序与 CAIPI recenter 参数。

Windows 权重哈希示例：

```powershell
Get-FileHash "C:\checkpoints\best_model.pt" -Algorithm SHA256
```

新训练保存的 best checkpoint 带 args；旧 checkpoint 缺少 args 时，必须从原运行命令/配置恢复，不能随意补元数据来“让它能加载”。

候选版 strict=True：结构不匹配就报错。这比只打印 missing/unexpected keys 后继续评估更容易发现错误权重。

下面仅是路径/参数模板，所有结构开关应按正式权重确认：

```powershell
python -m test.test --ckpt "C:\checkpoints\best_model.pt" --test_dir "E:\OCDI_data\test_prepared\measurements" --slice_ref_dir "E:\OCDI_data\test_prepared\slice_ref" --gt_dir "E:\OCDI_data\test_prepared\ground_truth" --out_dir "C:\OCDI_results\real_sample" --device cuda --coils 16 --base_ch 128 --T 64 --amp_norm --ref_lines 32 --kernel 9,9 --max_samples 1
```

若 checkpoint 没有结构元数据，需要补真实的 --xattn 或 --use_aux_k23 等开关。不要为了加载成功而随意开关。

至少对比：

1. 原始代码与候选代码对同一真实样本加载出的 K1/K2/K3/KMB、尺度、尺寸一致。
2. 相同权重、相同允许 ACS 和相同参数下的原/新预测与分数。
3. 去掉或扰动 GT 后预测一致。
4. 完整 held-out 集按最终协议运行，保留命令、日志、逐样本 CSV 与 hash。
5. 各基线对应的 ACS 与 split 检查。

候选版保存的是目标 recenter 后的复数 k-space，并记录 recenter_shift_frac。与其他输出比较前必须对齐坐标系、FFT、RSS、裁剪和指标后处理。

## 12. 在原地址替换完整文件树

先完成上述离线检查，并把正式需要的其他配置/Stage-U/基线相关材料补齐；不要把候选包直接描述成全论文最终版。

以下命令只在你自己的 Windows 电脑执行；本次没有替你删除或推送远端内容。

### 12.1 克隆、备份、建修复分支

```powershell
Set-Location "C:\"
git clone https://github.com/yqx7150/OCDI.git OCDI_upload
Set-Location "C:\OCDI_upload"
git bundle create "..\OCDI_before_packaging_fix.bundle" --all
git switch -c repair/packaging-acs
```

git bundle 保存 Git 提交历史，不包括未跟踪本地数据、GitHub Release 附件或外部权重。它们若存在，应另行保留。不要把备份 bundle 上传进代码仓库。

### 12.2 替换受跟踪文件

如果需要彻底替换当前受跟踪文件树，先确认候选目录存在且已验证：

```powershell
Test-Path "C:\OCDI_release_candidate\ocdi\data\packed.py"
Test-Path "C:\OCDI_release_candidate\ocdi\data\mb3.py"
```

两项都返回 True 后，在 C:\OCDI_upload 执行：

```powershell
git rm -r -- .
Get-ChildItem -LiteralPath "C:\OCDI_release_candidate" -Force |
    Where-Object { $_.Name -notin @(".git", ".venv", "build", "dist", "__pycache__") -and $_.Name -notlike "*.egg-info" } |
    Copy-Item -Destination "C:\OCDI_upload" -Recurse -Force
git add --all
git diff --cached --stat
git diff --cached --name-only
git ls-files ocdi/data
```

这里 git rm 只作用于本地受跟踪文件；.git 历史保留，公开 main 尚未变化。隐藏的 .gitignore 和 .github 必须一起复制。确认没有误删已有需要保留的 LICENSE、说明或其他配置；确认没有把真实数据、权重、访问凭证或个人路径文件一并加入。

若没有必须清除的旧文件，也可直接覆盖修正文件，不必执行 git rm。

### 12.3 在上传目录再次验收

重新安装该目录的代码，再跑数据测试与 smoke。不要误用另一个目录已安装的旧版本：

```powershell
Set-Location "C:\OCDI_upload"
python -m pip install -c constraints-cpu-tested.txt .
python -m pip check
python -m unittest discover -s tests -v
python -m scripts.smoke_test
git diff --cached --check
```

其中 python 应指向前面新建的验收环境。GPU 真实数据检查另用正式实验环境。

### 12.4 提交并推送修复分支

```powershell
git commit -m "Restore data package and document retrospective ACS protocol"
git push -u origin repair/packaging-acs
```

进入 GitHub 仓库页面，创建 repair/packaging-acs → main 的 Pull Request。检查 Files changed、Actions 日志和完整文件树，确认真实实验验证也已完成，再合并。

如果出现别人已经更新 main、分支保护或权限问题，按当前仓库规则处理；不要用强制推送覆盖别人的修改。

### 12.5 合并后重新克隆

```powershell
Set-Location "C:\"
git clone https://github.com/yqx7150/OCDI.git OCDI_final_check
Set-Location "C:\OCDI_final_check"
git rev-parse HEAD
git ls-files ocdi/data
```

在新环境按 README 重跑安装、入口、smoke 与一个真实样本。确认 ocdi/data、.github 和所有文档确实在远端。

最终稳定发布前，把 README 的 candidate 范围更新为已经真实核验的范围，完成 RELEASE_CHECKLIST。未完成项目继续标注，不删除事实性限制来制造完整的印象。

### 12.6 固定可引用版本

确认仓库没有同名 tag，且科学验证已完成后，例如使用：

```powershell
git tag -a v0.1.1 -m "Packaging repair and verified retrospective ACS interface"
git push origin v0.1.1
```

在 GitHub Releases 选择该 tag，写明实际修复与验证范围。若仍是候选版，应使用 RC tag 与 pre-release 标记，不把它标为完整验证的稳定版。

审稿回复给出固定提交和 tag 链接，同时保留原仓库地址。版本号只是示例，应与仓库现有版本对齐。

## 13. 论文与审稿回复建议

只有所有方法的历史实现核对通过后，才能将“都采用中心 ACS”写成已完成实验的事实。

建议采用的核心中文说明：

在回顾性比较中，各方法使用相同的多带观测和相同的允许校准信息。对于需要分层或面内参考的组件，仅利用相应单层 k-space 的中心 ACS 区域，回顾性模拟 slice_ref 和 ppa_ref；所有参考依赖的核校准及线圈敏感度估计均限制在该支持内。无需参考的组件不额外使用参考数据。这些数据是模拟校准参考，并非独立采集的参考扫描。完整测试单层数据用于离线观测生成和最终评分，不用于一般模型训练、权重选择、超参数选择或 ACS 外的测试校准；按扫描进行的校准仅访问声明的 ACS。

英文模板见 DATA_PROTOCOL.md。不要说 “all methods use identical reference files” 或 “all methods were trained using ACS” 来混淆全局训练与扫描特定校准。

对审稿意见 4，可以在实际修复、推送和验证后写：

We thank the reviewer for identifying this packaging issue. We have restored the missing data-loading modules, corrected the repository ignore rules and package configuration, and updated the installation and data-format documentation. We verified the documented entry points from a clean installation and added synthetic integration checks covering training, evaluation, and the separation of reconstruction inputs from scoring targets. We also clarified that, in the retrospective experiments, the reference-dependent components use slice-resolved calibration references simulated from the declared central ACS support, rather than independently acquired reference scans. The corrected version is available at [commit/tag URL].

若真实数据与所有基线还没有核验，不要把它们写成已验证；可以把回复限定为实际完成的打包修复。若新增真实样本验证，补一句准确描述其数据、权重与结果一致性即可。

## 14. 发布前最终应交付的证据

- 缺失源码确实被 Git 跟踪、存在于远端与安装包。
- 新环境的安装、help、smoke、Actions 记录。
- 真实 subject/volume split 与生成流程。
- 每个论文配置的 checkpoint hash、完整参数和代码提交。
- 各方法校准支持的代码与运行日志。
- 对应真实数据的不变性检查。
- 全部比较统一的指标定义与最终结果 CSV。
- 正确说明 SB 校准参考来源、可用性及额外开销。
- 对无法公开的数据提供合法获取方式与可运行的处理说明。

只有这些事实可核查，才能有依据地消除目前的具体质疑；任何版本都无法承诺“不会再有任何人质疑”。

## 官方与论文资料

- GitHub 删除文件与历史：https://docs.github.com/en/repositories/working-with-files/managing-files/deleting-files-in-a-repository
- fastMRI 官方仓库与数据说明：https://github.com/facebookresearch/fastMRI
- 单层参考与 SMS/GRAPPA 校准示例论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC4854329/

资料用于解释概念，不替代核查本研究实际执行的实验。
