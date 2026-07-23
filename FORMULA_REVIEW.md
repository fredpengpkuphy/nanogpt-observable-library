# 1821 个观测量公式复核

## 总结

- 共检查 1821 条公式，覆盖 22 个 reduction 类型。
- 每条公式现在都先定义观测对象 `X_t`，最后用 `y_t` 表示图表中的值。
- 49 条带 `center` 的公式单独显示中心化步骤。
- 2 条 temporal 公式逐步显示原始统计量、差分、EMA 或趋势。
- 所有公式仍严格对应 `observable_lib.py`，没有使用近似公式替代真实计算。

## 展示规则

```text
X_t = 当前模块在第 t 次观测时的张量
X_t^(1) = 对 X_t 进行的中心化等变换（如果存在）
y_t = 最终绘制在图表上的标量
```

复杂公式会把辅助量拆行。例如标准差先显示 `std(X_t)`，下一行再定义均值
`μ`；massive activation 先定义 RMS 为 `r`，再用 `r` 表示阈值。

## 22 类公式逐项结论

| Reduction | 数量 | 优化后的直观结构 | 复核结论 |
|---|---:|---|---|
| `mean` | 236 | `mean(X) = 元素总和 / N` | 正确 |
| `std` | 310 | `std(X)` 与均值 `μ` 分行显示 | 严格保留 `N-1` 分母 |
| `l1_norm` | 75 | `‖X‖₁ = 绝对值之和` | 正确 |
| `l2_norm` | 262 | `‖X‖₂ = 平方和开根号` | 正确 |
| `rms` | 75 | `RMS(X) = 平方平均后开根号` | 正确 |
| `max_abs` | 226 | `maxabs(X) = 最大绝对值` | 正确 |
| `sparsity` | 174 | `nearzero(X) = 接近零的元素比例` | 严格保留 `10⁻⁶` 阈值 |
| `positive_fraction` | 162 | `positiveFrac(X) = 正元素比例` | 正确 |
| `entropy` | 25 | 熵和归一化概率 `p̃` 分行显示 | 严格保留绝对值归一化和数值下限 |
| `spectral_norm` | 48 | `‖X‖₂ = σ_max(X)` | 正确 |
| `row_std_mean` | 48 | 行内标准差及行均值分行显示 | 正确 |
| `col_std_mean` | 48 | 列内标准差及列均值分行显示 | 正确 |
| `effective_rank` | 24 | 矩阵化、奇异值权重和有效秩分行显示 | 正确 |
| `attention_entropy_mean` | 12 | 每个 query 的注意力熵再取平均 | 正确 |
| `attention_entropy_min` | 12 | 最小注意力熵及其近零化简分行显示 | 公式正确，但指标受首 token 结构性影响 |
| `attention_sink_first_token` | 12 | later-query 对首 token 的平均注意力 | 正确 |
| `attention_sink_domination` | 12 | 各 key 平均注意力的最大值 | 公式正确，早期位置有结构性优势 |
| `attention_sink_ratio` | 12 | 首 token 注意力乘以序列长度 `T` | 公式正确，但不是 causal-uniform 归一化 |
| `activation_rate` | 12 | `activationRate(X) = 正激活比例` | 正确 |
| `massive_activation_peak_ratio` | 12 | 最大激活除以 RMS，RMS 单独定义 | 正确 |
| `massive_activation_outlier_fraction` | 12 | 超过 `3 × RMS` 的元素比例 | 正确 |
| `massive_neuron_fraction` | 12 | 峰值超过 `3 × RMS` 的神经元比例 | 正确 |

## Source 与变换检查

- `weight`、`grad`、`update`、activation、preactivation、logits、attention 和
  GELU activation 都使用独立的源符号，再统一命名为 `X_t`。
- 模块路径只在第一行出现一次，不再在同一公式中反复展开。
- `center` 严格按照实现沿 batch/sample 轴减均值，而不是误写成全局中心化。
- temporal 算子按照真实执行顺序展示，不再把多级时序链压缩成一个难以理解的长式子。

## 验证结果

- 1821 条公式全部成功生成。
- 22 个 reduction 类型均使用专门公式，没有通用 fallback。
- 每条公式都包含 source `X_t` 和最终值 `y_t`。
- 没有缺失公式、对象字符串泄漏或花括号不平衡。
- 完整 catalog 与两个 run manifest 的定义集合仍然一致。
