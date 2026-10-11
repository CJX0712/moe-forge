# ⚡ MoEForge · 混合专家锻造炉

> 纯 JavaScript 手写 **Mixture-of-Experts (MoE)** —— 可学习路由器 (Noisy Top-K 门控) + 多个专家 MLP + 负载均衡辅助损失。
> **零深度学习框架依赖**、**浏览器内训练**、**自动微分反向传播**、**可复现、可验证**。

[![零依赖](https://img.shields.io/badge/deps-0-brightgreen)](https://github.com/CJX0712/moe-forge)
[![不变量](https://img.shields.io/badge/self--checks-4%2F4-green)](https://github.com/CJX0712/moe-forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![作者](https://img.shields.io/badge/作者-晨星-gold)](https://github.com/CJX0712)

---

## 一、这是什么

混合专家 (MoE) 是现代大模型（**GPT-4、Mixtral 8×7B、LLaMA-MoE、DeepSeek-MoE**）的核心稀疏架构：
每个 token 只激活少数几个「专家」网络，从而在**不线性增加计算量**的前提下，把参数量扩展到千亿级。

本仓库在**单个 HTML 文件**内，从零手写整套 MoE 引擎，无需 PyTorch / TensorFlow，
所有张量运算、自动微分、Top-K 路由、负载均衡均由原生 JS 实现，且配备 **4 条可独立验证的不变量**。

## 二、核心思想

```
输入 x ──► 路由器(router: Wr, br) ──► logits ──► +噪声(训练) ──► softmax(p)
                                                            │
                                          Top-K 掩码(保留最大 k 个) ──► 门控权重 w (每行和=1)
                                                            │
           专家0: h=relu(x·W1+b1), o=h·W2+b2      ┐
           专家1: ...                              ├─► y = Σ_e  w[:,e] ⊙ o_e
           ...                                     ┘
                                                            │
           任务损失 MSE(y, target)  +  α·N·‖ f̄ ‖²   ← 负载均衡辅助损失
```

- **路由器 (router)**：`logits = x·Wr + br`，对 N 个专家打分。
- **Noisy Top-K 门控**：训练时对 logits 加高斯噪声（`randn × noise`），鼓励探索；取最大的 k 个专家，
  其余置 `-1e9` 后再 softmax，得到门控权重 `w`（每行严格和为 1）。
- **专家 (experts)**：每个专家是一个小 MLP `h=relu(x·W1+b1), o=h·W2+b2`，输出维度与任务一致。
- **最终输出**：`y = Σ_e scaleRows(o_e, w[:,e])`，即按门控权重对各专家输出加权求和。

## 三、数学核心

**门控权重（Top-K 软掩码）**

```
w_i,e = softmax( mask_i,e · logit_i,e ),   mask_i,e = 0 若 e∈TopK(i) 否则 -∞ 截断(-1e9)
Σ_e w_i,e = 1   (每行归一)
```

**负载均衡辅助损失（Shazeer et al., 2017；Fedus et al., 2021）**

```
f_e   = mean_i p_i,e                      // 专家 e 的平均路由概率（含噪声 logits 的 softmax）
aux   = N · α · Σ_e (f_e)²               // 当路由越均匀，f_e 越接近 1/N，aux 越小
total = MSE(y, target) + aux
```

该损失在训练时把「路由分布均匀」这一软约束直接写进梯度，等效于鼓励所有专家都被使用，
从而**抑制专家坍缩 (expert collapse / starvation)**。

## 四、四个不变量（自检 ✅）

| # | 不变量 | 含义 | 验证阈值 |
|---|--------|------|----------|
| ① | 解析梯度 == 中心差分 | 手写反向传播与数值梯度逐参数一致 | `maxRelErr < 1e-3` |
| ② | 门控权重每行和 == 1 | Top-K 软掩码后数值归一 | `maxDev < 1e-9` |
| ③ | 可学习 | 固定数据集训练后 loss 显著下降 | `l1 < 0.9·l0` |
| ④ | 负载均衡有效 | 加 aux 后专家利用率 ≥ 不加（对照） | 5/5 种子不退化 |

> ④ 是 MoE 的**灵魂验证**：对照「不加辅助损失」的基线，证明负载均衡项确实提升了专家利用率，
> 而非靠运气。运行 `node verify.js` 可复现全部四条。

## 五、如何运行

### 浏览器（交互式训练 + 可视化）
直接用浏览器打开 `moe-forge.html`：
- 左侧滑块调节 `N / k / d / 步数 / 学习率 / α / 噪声 / 种子`；
- 「🎯 训练」实时观察 loss 曲线、专家负载柱状图、专家特化色图；
- 「🧪 运行自检」一键跑出 4 条不变量。

### Node（无头验证，可复现）
```bash
node verify.js
```
输出 ①②③④ 四条判定，全绿即证明引擎正确。

## 六、与 forge 系列的关系

本作是「AI 大全」手写实现系列中的 **MoE 专题**，填补了此前 forge 矩阵在「稀疏激活 / 条件计算」方向的空白。
同系列遵循同一工程纪律：零依赖、单文件、浏览器内自检、不变量交叉验证、作者署名 **晨星**。

## 七、参考

- Shazeer et al. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (2017).
- Fedus et al. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (2021).
- Mistral AI. **Mixtral of Experts** (2024, 8×7B).
- Vaswani et al. **Attention Is All You Need** (2017) —— 同上 Transformer 主干。

## 八、许可与作者

- **License**: MIT © 晨星
- **Author**: 晨星 (Morning Star) · GitHub [@CJX0712](https://github.com/CJX0712)
- 零重写、集大成、闭环交付：本实现复用世界顶级 MoE 理论（Shazeer / Fedus / Mistral），
  以最小依赖、最高可读的手写代码呈现，并完整开源、可运行、可复现、可验证。
