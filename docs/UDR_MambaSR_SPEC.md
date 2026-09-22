# UDR-MambaSR：基于 RGB 不确定性与 Depth 可靠性的选择性残差超分方案

## 1. 目标与实验依据

任务为：

$$
\text{LR RGB}+\text{Depth}\rightarrow\text{HR RGB}
$$

RGB 为主模态，Depth
为辅助几何模态。唯一性能锚点是最终测试性能超过已经训练完成的 RGB-only
MambaIRv2 baseline。

当前结果：

  -----------------------------------------------------------------------------------------------
  Model                  Set5            Set14             B100         Urban100         Manga109
  ---------- ---------------- ---------------- ---------------- ---------------- ----------------
  RGB          32.7645/0.9029   29.0786/0.7928   27.8639/0.7468   27.3198/0.8215   31.8458/0.9239
  baseline

  GRS v3.0     32.7342/0.9026   29.0384/0.7924   27.8687/0.7469   27.3130/0.8216   31.8735/0.9240

  GTSS v3.1    32.6658/0.9012   28.9966/0.7909   27.8183/0.7451   27.0942/0.8156   31.6662/0.9214
  -----------------------------------------------------------------------------------------------

GTSS 相对 baseline 的 PSNR 变化分别为
-0.0987、-0.0820、-0.0456、-0.2256、-0.1796 dB，五库平均约 -0.1263
dB。因此 GTSS 的核心机制不能继续通过简单调参挽救。

## 2. GTSS 失败原因反推

### 2.1 Semantic scan 相邻不等于空间几何相邻

GTSS 使用：

$$
D^s=Gather(D,x_{sort})
$$

并定义：

$$
Q_t=|D_t^s-D_{t-1}^s|
$$

但 MambaIRv2 的 semantic routing
可能有意把空间距离较远、深度不同、但纹理或重复结构相似的 token
组织到同一序列。对于 SR，这种 non-local/self-similarity
建模本身具有价值。

因此利用 depth difference 抑制 semantic scan 上的状态传播，可能反而破坏
MambaIRv2 已经学习到的长程结构关系。Urban100 和 Manga109
的明显下降尤其支持这一风险判断。

结论：

$$
\boxed{\text{删除 scan-path depth transition}}
$$

### 2.2 小参数扰动不等于小功能扰动

GTSS 使用：

$$
dts'=dts+\beta G
$$

虽然 beta 很小，但 Delta 属于 SSM 核心状态动力学。扰动会在大量
token、多个 ASSM 和递归状态传播中累积。

因此：

$$
\boxed{\text{small parameter perturbation}\neq\text{small functional perturbation}}
$$

下一版不再让 Depth 修改 routing、sorting、Delta、A/B/C 或 selective
scan。

### 2.3 Depth 的角色需要重新定义

已有实验总体显示，只要 Depth 大范围改变 RGB
主干内部计算，就容易产生负迁移。因此 Depth 从 controller 改为：

$$
\boxed{\text{conditional residual expert}}
$$

研究问题由"Depth 怎样改变 RGB 主干"变为：

> RGB baseline 在哪里可能需要帮助，并且 Depth 在这些位置是否足够可信？

# 3. 新模型：UDR-MambaSR

**UDR-MambaSR：Uncertainty-Guided Depth Residual Mamba for RGB-Depth
Image Super-Resolution**

核心原则：

$$
RGB\rightarrow Original\ MambaIRv2
$$

Depth 不进入 MambaIRv2 内部状态计算，只在 RGB backbone
完成特征提取后提供一次 late residual correction。

Depth residual 只有在两个条件同时满足时才允许进入：

1.  RGB representation 存在较高 ambiguity；
2.  Depth 在该位置具有较高 reliability。

即：

$$
\boxed{RGB\ Ambiguous\land Depth\ Reliable\Rightarrow Depth\ Correction}
$$

## 4. 总体网络

``` text
LR RGB
   |
   v
Original MambaIRv2
   |
   +--------------------------+
   |                          |
   |                   Routing Probability
   |                          |
   |                          v
   |                   RGB Ambiguity U_RGB
   |                          |
   |                          v
   |                     Dual Gate G
   |                          ^
   |                          |
Depth --> Reliability -----> C_D
   |
   +--> Lightweight Geometry Encoder
                              |
                              v
                       Depth Residual Expert
                              |
Final RGB Feature --------> gated residual
          |                   |
          +--------- + -------+
                    |
                    v
              Original SR Head
                    |
                    v
                  HR RGB
```

ASSB1--ASSB6 完整保持原 MambaIRv2。

# 5. RGB Ambiguity Estimation

ASSM 已产生 routing logits $R_t$。Depth 不修改 routing，只读取 soft
probability：

$$
p_{t,k}=softmax(R_t)_k
$$

计算 normalized entropy：

$$
U_t=
-\frac{1}{\log K}
\sum_{k=1}^{K}
p_{t,k}\log(p_{t,k}+\epsilon)
$$

得到：

$$
U_t\in[0,1]
$$

$U_t$ 高表示 routing distribution 较模糊。

需要严格说明：它不是严格的 SR reconstruction uncertainty，应称为 **RGB
routing ambiguity / representation ambiguity proxy**。

建议从后半部分 ASSB4、ASSB5、ASSB6 聚合：

$$
U_{RGB}=\frac{U^{(4)}+U^{(5)}+U^{(6)}}{3}
$$

以减少单个 routing layer 的随机波动。

# 6. Depth Reliability Estimation

Depth robust normalization：

$$
D_N=
clip\left(
\frac{D-P_2(D)}
{P_{98}(D)-P_2(D)+\epsilon},
0,1
\right)
$$

计算：

$$
E_D=|\nabla D_N|,\qquad
E_R=|\nabla Y(I_{LR})|
$$

构造：

$$
Z=[E_R,E_D,|E_R-E_D|,E_RE_D]
$$

得到：

$$
C_D=\sigma(f_{GRE}(Z))
$$

其中 $C_D(x,y)\in[0,1]$。

GRE 保持轻量：

``` text
4 channels
 -> Conv 3x3, 4->16
 -> GELU
 -> Conv 3x3, 16->8
 -> GELU
 -> Conv 1x1, 8->1
 -> Sigmoid
 -> C_D
```

# 7. Dual-Condition Selective Gate

最终 gate：

$$
\boxed{G=U_{RGB}\odot C_D}
$$

三种典型情况：

-   RGB 确定 + Depth 可靠：$U\approx0,C\approx1\Rightarrow G\approx0$；
-   RGB 不确定 + Depth
    不可靠：$U\approx1,C\approx0\Rightarrow G\approx0$；
-   RGB 不确定 + Depth
    可靠：$U\approx1,C\approx1\Rightarrow G\approx1$。

因此辅助模态不是因为"存在"就参与，而是因为"主模态需要且辅助模态可信"才参与。

# 8. Lightweight Depth Residual Expert

Depth 输入：

$$
[D_N,E_D]
$$

轻量编码：

``` text
2 channels
 -> Conv 3x3: 2->32
 -> GELU
 -> DWConv 3x3: 32->32
 -> GELU
 -> PWConv 1x1: 32->32
 -> F_D
```

取 ASSB6 输出 RGB feature $F_R$ 作为 context：

$$
R_D=
P_{out}
\left(
DWConv(
Concat(LN(F_R),F_D)
)
\right)
$$

其中：

$$
R_D\in\mathbb R^{B\times C\times H\times W}
$$

Depth branch 不重新学习完整 RGB representation，而只预测
geometry-related residual correction。

# 9. 最终融合

$$
\boxed{
F_{refined}
=
F_R+
\alpha G\odot R_D
}
$$

限制最大干预：

$$
\alpha=\alpha_{max}\tanh(a)
$$

建议：

$$
\alpha_{max}=0.1
$$

并对 residual projection 最后一层采用小方差初始化，使模型初始接近 RGB
baseline。

# 10. 唯一融合位置

不再使用 ASSB2/4/6 多次融合。

唯一位置：

$$
\boxed{\text{ASSB6 后、conv\_after\_body 前}}
$$

``` text
conv_first
 -> ASSB1
 -> ASSB2
 -> ASSB3
 -> ASSB4
 -> ASSB5
 -> ASSB6
 -> UDR Residual Correction
 -> conv_after_body
 -> global residual
 -> upsampler
 -> SR RGB
```

这样整个 MambaIRv2 feature extraction 主过程不受 Depth 干扰。

# 11. 为什么不采用 Cross-Attention

目前实验的主要证据不是融合能力不足，而是 Depth intervention 很容易破坏强
RGB baseline。

Cross-attention 会进一步扩大 Depth 到 RGB
的信息通路。在当前证据下，更合理的是：

$$
\boxed{\text{weak + conditional + late residual}}
$$

而不是更强的 cross-modal interaction。

# 12. 训练策略

UDR 不再建议从随机初始化让 RGB 和 Depth 一起重新训练 500k。

## Phase A：Residual Expert Learning

加载已训练 RGB baseline checkpoint。

冻结：

-   conv_first；
-   ASSB1--ASSB6；
-   conv_after_body；
-   reconstruction head / upsampler。

只训练：

-   GRE；
-   Depth Residual Expert；
-   residual gate / alpha。

建议 50k--100k iterations，初始：

$$
LR_D=1\times10^{-4}
$$

该阶段是正式训练策略的一部分，其目标是强制 Depth 学习 baseline
尚未解决的 residual。

## Phase B：Low-LR Joint Fine-Tuning

随后解冻 RGB backbone，采用差分学习率：

$$
LR_D=1\times10^{-4}
$$

$$
LR_{RGB}=1\times10^{-5}
$$

即：

$$
LR_{RGB}\approx0.1LR_D
$$

继续联合 fine-tuning。

# 13. Loss

第一版只使用 baseline 一致的 L1：

$$
\boxed{
\mathcal L=\|I_{SR}-I_{HR}\|_1
}
$$

不增加 edge、depth、perceptual、GAN、frequency 或 consistency
loss，以便直接判断 UDR architecture 本身是否能够带来 fidelity 增益。

# 14. 学习目标的本质变化

此前相当于：

$$
RGB+Depth\rightarrow HR
$$

两个模态共同重新学习 reconstruction。

UDR 则首先保留：

$$
RGB\rightarrow SR_{base}
$$

随后让 Depth branch 通过最终 SR loss 学习：

$$
e=HR-SR_{base}
$$

中能够由可靠 geometry prior 修正的部分。

Depth 不再承担完整 SR，而只学习 baseline 的剩余误差。

# 15. 与 GRS / GTSS 对比

  --------------------------------------------------------------------------
  项目              GRS v3.0             GTSS v3.1         UDR
  ----------------- -------------------- ----------------- -----------------
  RGB backbone      被辅助机制介入       被 Delta 调制     完整保留

  Depth→RGB feature RAGA                 无                一次 late
                                                           residual

  Depth→routing     GCR                  无                无

  Depth→Delta       无                   有                无

  Depth→A/B/C       无                   无                无

  Depth reliability 有                   有                保留

  RGB ambiguity     无                   无                新增

  融合位置          ASSB2/4/6            ASSB2/4/6         ASSB6 后一次

  Depth 角色        feature/controller   state controller  conditional
                                                           residual expert

  baseline 干扰     中                   功能敏感          低
  --------------------------------------------------------------------------

# 16. 论文创新点

## 16.1 RGB Ambiguity-Aware Auxiliary Activation

利用 MambaIRv2 自身 routing distribution 构建 representation ambiguity
proxy：

$$
U_{RGB}
$$

Depth 不再无条件参与，而是在 RGB representation 模糊时才被激活。

## 16.2 Reliability-Aware Depth Guidance

通过 RGB-depth edge consistency 得到：

$$
C_D
$$

抑制 pseudo-depth 中不可信的几何信息。

## 16.3 Dual-Condition Residual Correction

$$
G=U_{RGB}C_D
$$

$$
F'=F+\alpha GR_D
$$

形成：

$$
\boxed{
RGB\ ambiguity
+
Depth\ reliability
\rightarrow
Selective\ residual\ correction
}
$$

把多模态融合从无条件 feature interaction 转化为需求驱动的辅助模态激活。

# 17. 最终测试

测试集：

-   Set5
-   Set14
-   B100
-   Urban100
-   Manga109

主要比较：

$$
v1.0\ RGB
\quad vs.\quad
v3.0\ GRS
\quad vs.\quad
v3.1\ GTSS
\quad vs.\quad
UDR
$$

考虑原 MambaIRv2 routing 的推理随机性，最终 baseline 与 UDR 建议各重复 3
次 inference，报告 mean ± std；无需为此增加新的模型训练。

# 18. 成功标准

最低：

$$
\boxed{Avg\Delta PSNR>0}
$$

具有进一步论文价值：

$$
\boxed{Avg\Delta PSNR\gtrsim+0.05\text{ dB}}
$$

同时重点观察 Urban100 与 Manga109，希望至少一个结构复杂数据集获得约
+0.05--+0.10 dB 或以上的稳定提升。

# 19. Depth 信息上限

如果当前 Depth 来自 monocular depth estimator：

$$
D=f(RGB)
$$

那么它并不是独立传感器信息。

Depth 的潜在价值主要是：

$$
\boxed{\text{将 RGB 中隐式 geometry 重新参数化为显式先验}}
$$

而不是提供大量新的 HR texture。

因此 UDR 把 Depth 定位为小幅 correction，而不是第二主干。

# 20. 明确停止规则

如果 UDR 完整训练后仍满足：

$$
Avg\Delta PSNR\le0
$$

且 Urban100、Manga109 均没有稳定正增益，则不建议继续设计新的：

-   Depth-Mamba routing；
-   Depth selective scan；
-   Cross-Mamba；
-   多层 Cross-Attention；
-   更深 Depth backbone。

此时更符合现有实验事实的判断是：

$$
\boxed{
\text{当前 Depth 数据对 bicubic ×4 RGB fidelity-SR 强 baseline 的可利用增量信息不足}
}
$$

后续应优先重新考虑 Depth 来源与质量、是否具有真正独立 RGB-D
observation、辅助模态类型、数据集/退化模型和 multimodal SR 的评价目标。

# 21. 最终总结

GTSS 的失败意味着：

$$
\cancel{Depth\rightarrow scan\ transition\rightarrow\Delta}
$$

不应继续。

UDR 最终采用：

$$
\boxed{
RGB\rightarrow Original\ MambaIRv2
\rightarrow U_{RGB}
}
$$

$$
\boxed{
Depth\rightarrow C_D,F_D
}
$$

$$
\boxed{
G=U_{RGB}\odot C_D
}
$$

$$
\boxed{
F_{refined}=F_R+\alpha G\odot R_D
}
$$

核心原则：

> **Depth 不再告诉 MambaIRv2 应该怎样计算，而是在 RGB
> 主干完成表征之后，仅当 RGB 表示存在 ambiguity 且 Depth
> 本身可靠时，提供受严格限制的 late residual correction。**

这是根据此前多版 RGB+Depth 模型以及 GRS v3.0、GTSS v3.1
的真实实验结果，对当前多模态 SR 路线做出的进一步收缩和修正。
