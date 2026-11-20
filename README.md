# Cerebellum-inspired Spiking Neural Network

A spiking network modelled on the cerebellar cortex, trained on MNIST with **local learning rules only — no backpropagation and no autograd**.

**Test accuracy 89.08%** · 73,728 synapses · 73,738 plastic weights · T = 25 time steps

---

## 1. 架构

```
MF（784 个像素，direct 编码：像素值作为恒定输入电流，不随时间变化）
  │  固定稀疏连接，每单元 fan-in 8，w ≥ 0，Σw = 1，**永不训练**
  ▼
GrC（4096 个 LIF 神经元）  ◄── GoC：16 组 × 256 单元，低通群活动 → 减法式反馈抑制
  │
  ▼
PC（10 类 × 1 = 10 个 LIF 神经元）   ← **唯一的 CF 教导可塑位点（PF→PC）**
  │  抑制性输出（GABA 能）
  ▼
DCN（静态读出）：  logit_c = b_c − γ · I_c
                   I_c = (1/T) Σ_t [ (1/G) Σ_{i∈PC(c)} s_i[t] ]²
```

### 1.1 四个群体

| 群体 | 数量 | 神经元模型 | 作用 |
|---|---|---|---|
| **MF** | 784 | 输入电流 | MNIST 像素直接作为恒定电流 |
| **GrC**（颗粒细胞） | 4096 | LIF，β=0.9，θ=1.0 | 固定随机展开；形成稀疏、高维编码 |
| **GoC**（高尔基细胞） | 16 组 × 256 | 比例控制器 | 局部反馈抑制，把 GrC 活动率压到 5% |
| **PC**（浦肯野细胞） | 10（每类 1 个） | LIF，β=0.9，θ=1.0 | 唯一被教导信号修改的位点 |
| **DCN**（深部小脑核） | 10（每类 1 个） | 静态读出 | 把 PC 的抑制转成类别分数 |

### 1.2 逐层连接

| 连接 | 形状 | 性质 | 可塑 |
|---|---|---|---|
| MF → GrC | 4096 × 8 = **32,768** | 兴奋，固定稀疏，行和 = 1 | ❌ |
| GrC → GoC | 16 组（组内均值） | 兴奋 | ❌ |
| GoC → GrC | 4096（每组一个增益） | **抑制**，减法 | ❌ |
| GrC → PC | 10 × 4096 = **40,960** | 兴奋，行和 = 1 | ✅ **CF 门控** |
| PC → DCN | 10 × 1 | **抑制**，γ = 3.0 | ❌（固定标量） |
| MF → DCN | 10（tonic 偏置 `b_c`） | 兴奋 | ✅ 误差驱动 |
| **总突触** | **73,728** | | |

### 1.3 读出公式

```
logit_c = b_c − γ · I_c        γ = 3.0
I_c     = (1/T) Σ_t ( mean_{i∈PC(c)} s_i[t] )²
```

**注意符号**：PC 是抑制性的，所以 **PC 活动越强，该类的分数越低**。分类等价于"选哪个类的 PC 最安静"。

> 上式中的 `(·)²` 是超线性池化，只在"每类多个 PC"时才改变行为。本配置 `G = 1`，所以它**退化为线性**，`I_c` 就是该类 PC 的平均发放率。

### 1.4 规模

| 项 | 值 |
|---|---|
| 时间步 T | 25 |
| `nn.Parameter` 数量 | 40,971 |
| 被局部规则修改的权重 | **73,738**（PF→PC 40,960 + MF→GrC 32,768 + DCN 偏置 10） |
| 前向是否用 autograd | **否** |
| 反向是否用 autograd | **否**（全部规则手写） |

---

## 2. 训练机制

**核心：四条学习规则，全部是局部规则，没有一次反向传播。**

### 2.1 PF→PC：CF 门控的多类感知机（核心规则）

$$\Delta w_{c,j} = -\eta\,\mathbb{1}[c=y,\ \hat y\neq y]\cdot\hat z_j \;+\; \eta\,\mathbb{1}[c=\hat y\neq y]\cdot\hat z_j$$

其中 $\hat z = z/\lVert z\rVert_1$（`z` 是 GrC 的归一化脉冲计数）。每次更新后把每个 PC 的行归一化回 $\sum_j w = 1$。

| 情形 | 需要 | 动作 | 机制 |
|---|---|---|---|
| $c = y$ 但判错 | 提高 $\text{logit}_c$ | 削弱 $w_c$（LTD） | PC 发放 ↓ → 抑制减弱 → 分数升 |
| $c = \hat y \neq y$ | 降低 $\text{logit}_c$ | 加强 $w_c$（LTP） | PC 发放 ↑ → 抑制增强 → 分数降 |

- **只在判错时更新**；判对完全不动。
- **$\eta$ 是无量纲的**：一行权重是 4096 个 GrC 上的概率分布，$\eta$ 表示"这一步把该 PC 权重分布的多少比例质量搬向/搬离本次的活跃模式"。$\eta = 0.3$ 即每步搬 30%，所以最优区间在 0.1–0.5。
- **行归一化是规则的一部分**，不是装饰：它把 LTD 变成对比式更新（削弱 $w_y$ 后重新归一 = 等价于抬高其余所有位置）。

### 2.2 PC 内在可塑性

每个 PC 一个乘性增益 `pc_h`，用比例控制器把发放率钉在 0.10：

```
ratio  = rate_target / pc_rate          # 0.10 / 实测
pc_h  *= ratio ** eta_h                 # eta_h = 0.5，限幅 [0.05, 20]
```

不加这一条，单向规则会让部分 PC 静默、部分饱和（实测三步后 0.009 vs 0.302）。这是**与标签无关**的控制器，每个 batch 都运行。

### 2.3 GoC：稀疏性控制器

```
ratio = grc_rate_per_group / goc_target     # 目标 5%
goc_g *= ratio ** eta_goc                   # 比例控制，限幅 [0.2, 200]
```

**16 个组各有一个独立增益**（不是全局标量）。用比例而非积分控制：积分器在单次 T=25 前向内不可能收敛（实测活动率会停在 37.5%）。

### 2.4 GoC 门控 MF→GrC 可塑性

MF→GrC 权重用**无监督**的竞争/Oja 规则学习（无标签、无 CF）：

```
xb    = x[:, idx] / Σ x[:, idx]                 # 每单元自身扇入上的相对模式
num   = Σ_b s_bj · xb_bj
den   = Σ_b s_bj
Δw    ∝ num − den · w                           # Oja 式方向
```

并由 GoC 状态决定**更新方向**（BCM 式滑动阈值）：

```
gate   = tanh((goc_target − grc_rate_ema) / gate_scale)
factor = 1 + gate_kappa · gate                  # gate_kappa = 1.0
Δw    *= factor
w      = normalize(w)                           # 回到 simplex
```

依据：Golgi 回路完整时 MF→GrC 以 LTD 为主，阻断 GABA-A 后翻转为 LTP——**抑制状态本身、而不只是"该细胞有没有发放"，决定可塑性符号**。

### 2.5 一次性训练流程

```
for epoch in 1..30:
    for batch in train_loader:                  # batch = 128
        ① 前向（no_grad）：得到 GrC 计数、PC 计数、logits
        ② grc_update        ← 无监督更新 MF→GrC
        ③ goc_homeo_update  ← 更新 16 个 GoC 增益
        ④ cave_update       ← CF 门控更新 PF→PC、PC 内在增益、DCN 偏置
    evaluate(val) 并保存 val 最高的检查点
```

**整个循环里没有 `loss.backward()`。**

### 2.6 完整超参

| 参数 | 值 | | 参数 | 值 |
|---|---|---|---|---|
| `n_grc` | 4096 | | `eta0`（LTD） | 0.3 |
| `grc_fan_in` | 8 | | `eta1`（LTP） | 0.3 |
| `n_goc_groups` | 16 | | `groups` | `bidir` |
| `n_pc_per_class` | 1 | | `eta_h` | 0.5 |
| `T` | 25 | | `rate_target` | 0.10 |
| `beta` / `thresh` | 0.9 / 1.0 | | `grc_learn` | `hebb` |
| `encoder` | `direct` | | `eta_g` | 0.1 |
| `pc_gain` | 5.0 | | `goc_homeo` | True |
| `goc_gain` / `goc_tau` | 2.0 / 2.0 | | `eta_goc` / `goc_target` | 0.5 / 0.05 |
| `gamma` | 3.0 | | `goc_gate` | True |
| `epochs` | 30 | | `gate_kappa` / `gate_scale` | 1.0 / 0.02 |
| `batch_size` | 128 | | `seed` | 0 |

### 2.7 复现

```bash
pip install -r requirements.txt

# 直接评估随仓库附带的权重（不训练，几秒）
python eval.py --ckpt checkpoint/best.pt
# -> accuracy : 89.08%

# 从零训练（约 17 分钟 GPU / 更久 CPU）
python train.py --epochs 30 --goc-homeo --grc-learn hebb --eta-g 0.1 --goc-gate

# 七项不变量自检（不需要训练数据）
python model.py
```

数据：首次运行会自动把 MNIST 下载到本目录的 `data/`。

---

## 3. Benchmark 报告数据

### 3.1 主结果

| 指标 | 值 |
|---|---|
| **MNIST test accuracy** | **89.08%** |
| best validation accuracy | 88.12%（epoch 28） |
| 训练 / 验证 / 测试划分 | 55,000 / 5,000 / 10,000 |
| 训练轮数 | 30 |
| 训练耗时 | 997.8 秒（≈16.6 分钟） |
| 训练后 GrC 活动率 | 5.13% |
| 训练后 PC 发放率 | 10.56% |

**测试集标准误 ≈ 0.31 pt，验证集（5k）标准误 ≈ 0.46 pt。**

### 3.2 训练曲线（节选；完整 30 行见 `checkpoint/metrics.csv`）

| epoch | train acc | val acc | GrC 率 | PC 率 | 错分数 |
|---|---|---|---|---|---|
| 1 | 72.84% | 79.62% | 0.0504 | 0.0973 | 14,937 |
| 5 | 84.64% | 86.52% | 0.0500 | 0.1048 | 8,449 |
| 10 | 86.00% | 83.04% | 0.0500 | 0.1024 | 7,701 |
| 15 | 86.59% | 87.84% | 0.0500 | 0.1084 | 7,376 |
| 20 | 86.60% | 86.42% | 0.0500 | 0.1057 | 7,372 |
| **28** | **86.91%** | **88.12%** ← 选中 | 0.0500 | 0.1019 | 7,198 |
| 30 | 86.89% | 82.84% | 0.0500 | 0.1051 | 7,210 |

验证集在训练中抖动较大（epoch 10 / 15 / 18 分别为 83.04 / 87.84 / 84.34），因此"选哪个 epoch"对最终数字有实质影响。

### 3.3 与参考线的对比

| 参考线 | test acc | 性质 |
|---|---|---|
| **本模型** | **89.08%** | 73,738 个局部规则更新的权重 |
| 无约束线性读出（signed ridge，冻结 GrC 编码） | 96.06% | 编码端线性可解码上界 |
| 同突触预算 + 梯度训练前端（T4 flat，74,000 突触） | 97.85% | 同预算 BP 对照 |
| 非负权重 NNLS（闭式拟合） | 79.11% | 下界估计（不是上界） |
| 随机猜测 | 10.00% | |

- 与**同预算 BP 对照**差 **8.77 pt**。
- 与**编码端线性上界**差 **6.98 pt**。

### 3.4 消融

| 配置 | test acc | 相对主结果 |
|---|---|---|
| **完整模型** | **89.08%** | — |
| 关掉 CF 教导信号（`--no-cf`） | **9.53%** | −79.55 pt → **随机水平** |
| 关掉 GoC（`--goc-gain 0`） | 70.42% | −18.66 pt |
| 每类 2 个 PC（`--n-pc-per-class 2`） | 80.17% | −8.91 pt |
| 双组集成（`--n-group 2 --groups bidir_all`） | 79.65% | −9.43 pt |
| 只开 GoC 自稳（无 GrC 学习） | 77.86% | −11.22 pt |

**关掉教导信号即回落到随机水平**，说明这套架构的学习完全依赖攀爬纤维携带的误差信号。

### 3.5 同一架构的不同版本

| 版本 | test acc | 差异 |
|---|---|---|
| **本版本**（per-group GoC 增益 + GoC 门控） | **89.08%** | — |
| per-group GoC 增益，不门控 | 89.02% | −0.06 pt（噪声内） |
| 全局标量 GoC 增益 | 88.44% | −0.64 pt |
| 最早可用版本 | 83.76% | −5.32 pt |

> per-group 增益相对全局标量的增益为 **+0.58 pt（≈1.9 倍标准误）**，处于边缘；门控机制额外贡献的 **+0.06 pt** 落在噪声内。

---

MIT License — see [LICENSE](LICENSE).
