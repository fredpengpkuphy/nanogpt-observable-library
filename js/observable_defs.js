/**
 * Direct mathematical formulas for each observable (source ⊗ transform* ⊗ reduction ⊗ temporal*).
 * KaTeX bodies without surrounding $$.
 */

const TRANSFORM_DEFS = {
  identity: true,
  center: true,
  abs: true,
  normalize: true,
  square: true,
};

const TEMPORAL_NAMES = {
  identity: true,
  delta: true,
  ema: true,
  slope: true,
  rolling_std: true,
  curvature: true,
};

function parseOpName(raw) {
  const m = String(raw).match(/^([a-zA-Z_][a-zA-Z0-9_]*)/);
  return m ? m[1] : String(raw);
}

function parseOpArgs(raw) {
  const m = String(raw).match(/^[a-zA-Z_][a-zA-Z0-9_]*\((.*)\)$/);
  return m ? m[1] : "";
}

function parseNamedArg(args, key) {
  if (!args) return null;
  const re = new RegExp(`(?:^|,)\\s*${key}\\s*=\\s*([^,\\s)]+)`, "i");
  const m = args.match(re);
  return m ? m[1] : null;
}

function escapeTexPath(sel) {
  return String(sel || "")
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([#%&{}])/g, "\\$1")
    .replace(/_/g, "\\_");
}

function splitPipelineOps(transforms) {
  const tensorOps = [];
  const temporalOps = [];
  for (const raw of transforms || []) {
    const name = parseOpName(raw);
    if (TEMPORAL_NAMES[name] && !TRANSFORM_DEFS[name]) temporalOps.push(raw);
    else if (TRANSFORM_DEFS[name] || name === "identity") {
      if (name !== "identity") tensorOps.push(raw);
    } else if (TEMPORAL_NAMES[name]) temporalOps.push(raw);
    else tensorOps.push(raw);
  }
  return { tensorOps, temporalOps };
}

/** Source tensor as a direct call with module path. */
function sourceCallTex(spec) {
  const path = escapeTexPath(spec.selector || spec.ui_module || "?");
  const kind = spec.source_kind || "tensor";
  const map = {
    weight: "weight",
    grad: "grad",
    update: "update",
    opt_m: "opt\\_m",
    opt_v: "opt\\_v",
    activation: "act",
    preactivation: "preact",
    logits: "logits",
    attention: "attn",
    gelu_activation: "gelu",
  };
  const name = map[kind] || kind.replace(/_/g, "\\_");
  return `\\mathrm{${name}}(\\mathtt{${path}})`;
}

function applyTransformTex(op, x) {
  const name = parseOpName(op);
  if (name === "center") return `\\big(${x}-\\mathrm{mean}(${x})\\big)`;
  if (name === "abs") return `\\lvert ${x}\\rvert`;
  if (name === "normalize") return `\\dfrac{${x}}{\\|${x}\\|_2+\\varepsilon}`;
  if (name === "square") return `(${x})^{2}`;
  return x;
}

/**
 * Wrap transformed tensor x with the reduction → scalar expression.
 * Attention / massive activations use specialized closed forms.
 */
function applyReductionTex(reduction, x) {
  switch (reduction) {
    case "mean":
      return `\\mathbb{E}[${x}]`;
    case "std":
      return `\\mathrm{Std}(${x})`;
    case "min":
      return `\\min ${x}`;
    case "max":
      return `\\max ${x}`;
    case "l1_norm":
      return `\\|${x}\\|_1`;
    case "l2_norm":
      return `\\|${x}\\|_2`;
    case "rms":
      return `\\mathrm{RMS}(${x})`;
    case "max_abs":
      return `\\max_i\\lvert ${x}_i\\rvert`;
    case "sparsity":
      return `\\dfrac{1}{N}\\sum_i\\mathbf{1}\\big[\\lvert ${x}_i\\rvert<10^{-6}\\big]`;
    case "positive_fraction":
      return `\\dfrac{1}{N}\\sum_i\\mathbf{1}[${x}_i>0]`;
    case "entropy":
      return `-\\sum_i p_i\\log p_i,\\; p_i=\\dfrac{\\lvert ${x}_i\\rvert}{\\sum_j\\lvert ${x}_j\\rvert}`;
    case "spectral_norm":
    case "top_singular_value":
      return `\\sigma_{\\max}(${x})`;
    case "trace":
      return `\\mathrm{tr}(${x})`;
    case "row_std_mean":
      return `\\dfrac{1}{r}\\sum_{i=1}^{r}\\mathrm{Std}((${x})_{i,:})`;
    case "col_std_mean":
      return `\\dfrac{1}{c}\\sum_{j=1}^{c}\\mathrm{Std}((${x})_{:,j})`;
    case "effective_rank": {
      return (
        `\\exp\\!\\Big(-\\sum_k p_k\\log p_k\\Big),\\;` +
        `p_k=\\dfrac{\\lambda_k}{\\sum_j\\lambda_j},\\;` +
        `\\lambda_k=\\sigma_k(${x}-\\overline{${x}})^{2}`
      );
    }
    case "attention_entropy_mean":
      return (
        `\\mathbb{E}_{b,h,q}\\Big[-\\sum_k ${x}_{b,h,q,k}\\log ${x}_{b,h,q,k}\\Big]`
      );
    case "attention_entropy_min":
      return (
        `\\min_{b,h,q}\\Big(-\\sum_k ${x}_{b,h,q,k}\\log ${x}_{b,h,q,k}\\Big)`
      );
    case "attention_sink_first_token":
      return `\\mathbb{E}_{b,h,q\\ge 1}[${x}_{b,h,q,0}]`;
    case "attention_sink_domination":
      return `\\max_k\\mathbb{E}_{b,h,q}[${x}_{b,h,q,k}]`;
    case "attention_sink_ratio":
      return `T\\cdot\\mathbb{E}_{b,h,q\\ge 1}[${x}_{b,h,q,0}]`;
    case "activation_rate":
      return `\\dfrac{1}{N}\\sum_i\\mathbf{1}[${x}_i>0]`;
    case "massive_activation_peak_ratio":
      return `\\dfrac{\\max\\lvert ${x}\\rvert}{\\mathrm{RMS}(${x})}`;
    case "massive_activation_outlier_fraction":
      return `\\dfrac{1}{N}\\sum_i\\mathbf{1}\\big[\\lvert ${x}_i\\rvert>3\\,\\mathrm{RMS}(${x})\\big]`;
    case "massive_neuron_fraction":
      return (
        `\\dfrac{1}{F}\\sum_f\\mathbf{1}\\big[\\max_{b,t}\\lvert ${x}_{b,t,f}\\rvert>3\\,\\mathrm{RMS}(${x})\\big]`
      );
    default: {
      const safe = String(reduction || "R").replace(/[^a-zA-Z0-9_]/g, "");
      return `\\mathcal{R}_{\\mathrm{${safe}}}(${x})`;
    }
  }
}

function wrapTemporal(raw, scalarTex) {
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "identity") return `y_t=${scalarTex}`;
  if (name === "delta") return `y_t=(${scalarTex})-(${scalarTex})_{t-1}`;
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    return `y_t=${a}\\,y_{t-1}+(1-${a})\\,(${scalarTex})`;
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `y_t=\\mathrm{OLS\\,slope}\\big(\\{(${scalarTex})_{t-${w}+1},\\ldots,(${scalarTex})_t\\}\\big)`;
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `y_t=\\mathrm{Std}\\big((${scalarTex})_{t-${w}+1},\\ldots,(${scalarTex})_t\\big)`;
  }
  if (name === "curvature") {
    return `y_t=(${scalarTex})_t-2(${scalarTex})_{t-1}+(${scalarTex})_{t-2}`;
  }
  return `y_t=${scalarTex}`;
}

/* ---------- Plain-language “what this measures” (Chinese) ---------- */

function modulePlaceZh(spec) {
  const role = spec.role || "";
  const sel = spec.selector || "";
  const layer = spec.layer;
  const L = layer === null || layer === undefined ? null : Number(layer);

  const layerPrefix = L == null ? "" : `第 ${L} 层 `;

  if (role === "wte" || sel.includes("wte")) return "Token Embedding（wte）";
  if (role === "wpe" || sel.includes("wpe")) return "Position Embedding（wpe）";
  if (role === "ln_f" || sel.includes("ln_f")) return "最终 LayerNorm（ln_f）";
  if (role === "lm_head" || sel.includes("lm_head")) return "LM Head（词表 logits 投影）";

  if (role === "ln_1" || /\.ln_1/.test(sel)) return `${layerPrefix}Pre-Attention LayerNorm（ln_1）`;
  if (role === "ln_2" || /\.ln_2/.test(sel)) return `${layerPrefix}Pre-MLP LayerNorm（ln_2）`;
  if (role === "attn.c_attn" || sel.includes("c_attn")) {
    return `${layerPrefix}Attention QKV 投影（c_attn）`;
  }
  if (role === "attn.c_proj" || /attn\.c_proj/.test(sel)) {
    return `${layerPrefix}Attention 输出投影（c_proj）`;
  }
  if (
    role === "attn" ||
    /^h\.\d+\.attn$/.test(spec.ui_module || "") ||
    (sourceLooksLikeAttentionWeights(spec) && !sel.includes("c_attn") && !sel.includes("c_proj"))
  ) {
    return `${layerPrefix}因果 Self-Attention（softmax 注意力权重）`;
  }
  if (role === "mlp.c_fc" || sel.includes("c_fc")) {
    return `${layerPrefix}MLP 上投影（c_fc）`;
  }
  if (role === "mlp.c_proj" || /mlp\.c_proj/.test(sel)) {
    return `${layerPrefix}MLP 下投影（c_proj）`;
  }
  if (role === "mlp.gelu" || (spec.ui_module || "").includes("gelu") || spec.source_kind === "gelu_activation") {
    return `${layerPrefix}MLP GELU 后隐层`;
  }

  if (spec.ui_module) return `${layerPrefix}${spec.ui_module}`;
  return sel || "该模块";
}

function sourceLooksLikeAttentionWeights(spec) {
  return spec.source_kind === "attention" || String(spec.reduction || "").startsWith("attention_");
}

function sourceObjectZh(spec) {
  switch (spec.source_kind) {
    case "weight":
      return "参数权重张量 θ";
    case "grad":
      return "反向传播得到的梯度 g=∇_θℒ";
    case "update":
      return "相邻观测步之间的参数更新量 u=θ_t−θ_{t−1}";
    case "opt_m":
      return "Adam 一阶矩估计 m̂（exp_avg）";
    case "opt_v":
      return "Adam 二阶矩估计 v̂（exp_avg_sq）";
    case "activation":
      return "该 Linear 的前向输出激活 a";
    case "preactivation":
      return "该 Linear 的输入（激活前）ã";
    case "logits":
      return "LM head 输出的 logits z（对词表维）";
    case "attention":
      return "因果 softmax 注意力矩阵 A∈ℝ^{B×H×T×T}（query→key 概率）";
    case "gelu_activation":
      return "MLP 中 GELU 之后的隐层激活";
    default:
      return spec.source_kind || "张量";
  }
}

function transformClauseZh(tensorOps) {
  if (!tensorOps?.length) return "";
  const parts = [];
  for (const op of tensorOps) {
    const name = parseOpName(op);
    if (name === "center") {
      parts.push("先沿 batch（sample）维减去均值；若无 batch 维则对全体元素去均值");
    } else if (name === "abs") {
      parts.push("先取绝对值");
    } else if (name === "normalize") {
      parts.push("先沿 feature 维做 L2 归一化（无 feature 维则整体归一化）");
    } else if (name === "square") {
      parts.push("先对元素平方");
    } else {
      parts.push(`先施加变换 ${name}`);
    }
  }
  return parts.length ? parts.join("，") + "，然后" : "";
}

function reductionMeaningZh(reduction, sourceKind) {
  const isAttn = sourceKind === "attention" || String(reduction).startsWith("attention_");
  const isGelu = sourceKind === "gelu_activation" || String(reduction).startsWith("massive_") || reduction === "activation_rate";

  switch (reduction) {
    case "mean":
      return "全体元素的均值，反映整体偏置/平均水平";
    case "std":
      return "全体元素的标准差，反映离散程度与尺度波动";
    case "min":
      return "全体元素的最小值";
    case "max":
      return "全体元素的最大值";
    case "l1_norm":
      return "L1 范数（元素绝对值之和），衡量总“质量”/稀疏相关的尺度";
    case "l2_norm":
      return "L2 范数（欧氏长度），衡量该张量的总体能量/尺度";
    case "rms":
      return "均方根 RMS=√E[x²]，衡量典型幅度（对符号不敏感）";
    case "max_abs":
      return "最大绝对值，捕捉最极端的单个元素幅度";
    case "sparsity":
      return "近似零元素比例（|x|<10⁻⁶），衡量有多少分量接近失活/为零";
    case "positive_fraction":
      return "正元素所占比例，反映符号偏置与“激活”占比（对权重/梯度/激活均适用）";
    case "entropy":
      return "把 |x| 归一化成概率质量后的 Shannon 熵：越高说明质量越分散，越低说明集中在少数分量上";
    case "spectral_norm":
    case "top_singular_value":
      return "权重矩阵的最大奇异值 σ_max，衡量该线性映射的最大增益（算子范数）";
    case "trace":
      return "方阵的迹 tr(W)，即对角元之和（非方阵时无定义）";
    case "row_std_mean":
      return "各输出行（out）标准差的均值，衡量行方向上的异质性";
    case "col_std_mean":
      return "各输入列（in）标准差的均值，衡量列方向上的异质性";
    case "effective_rank":
      return "基于去中心化后协方差特征值熵的有效秩：越大说明能量分布在更多方向上，越小说明低秩/坍缩";
    case "attention_entropy_mean":
      return isAttn
        ? "每个 (batch, head, query) 对 key 分布的 Shannon 熵再取平均：越高注意力越分散，越低越尖锐集中"
        : "注意力熵的均值";
    case "attention_entropy_min":
      return "所有 (batch, head, query) 上注意力熵的最小值：越小说明存在极度 peaked 的注意力";
    case "attention_sink_first_token":
      return "非首 token 的 query 分配给 key=0（序列首位置）的平均注意力质量，衡量 attention sink 强度";
    case "attention_sink_domination":
      return "各 key 位置收到的平均注意力质量的最大值：接近 1 表示几乎所有质量汇到单一 key（强 sink）";
    case "attention_sink_ratio":
      return "首 token sink 质量相对均匀注意力 1/T 的倍数（= T·E[A_{q,0}]）：>1 表示首位置被过度关注";
    case "activation_rate":
      return isGelu
        ? "GELU 后隐层中 x>0 的元素比例（firing rate），衡量有多少单元处于“激活”侧"
        : "x>0 的元素比例（firing rate）";
    case "massive_activation_peak_ratio":
      return "max|x| / RMS(x)：越大说明存在相对背景被强烈放大的 massive activation 尖峰";
    case "massive_activation_outlier_fraction":
      return "|x|>3·RMS(x) 的元素占比，衡量逐元素 outlier / massive activation 的覆盖面";
    case "massive_neuron_fraction":
      return "在 hidden 维上，若某神经元在全部 batch×time 上的峰值 max|x| 超过 3·RMS(全局)，则计为 massive neuron；本指标是这类神经元占比";
    default:
      return `标量汇总「${reduction}」`;
  }
}

function temporalClauseZh(temporalOps) {
  if (!temporalOps?.length) return "";
  const raw = temporalOps[0];
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "delta") return "再对训练步序列做一阶差分（相对上一观测步的变化）。";
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    return `再对训练步序列做 EMA 平滑（α=${a}）。`;
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `再在最近 ${w} 个观测点上做 OLS 斜率，看短期趋势。`;
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `再取最近 ${w} 个观测点的滚动标准差，看短期波动。`;
  }
  if (name === "curvature") return "再做三点二阶差分，看加速度/曲率。";
  return "";
}

/**
 * Direct plain-language description: what this observable is measuring.
 */
function buildSpecPlainDescription(spec) {
  if (!spec) return "";
  const place = modulePlaceZh(spec);
  const obj = sourceObjectZh(spec);
  const { tensorOps, temporalOps } = splitPipelineOps(spec.transforms || []);
  const tClause = transformClauseZh(tensorOps);
  const meaning = reductionMeaningZh(spec.reduction, spec.source_kind);
  const temporal = temporalClauseZh(temporalOps);

  const core = `测的是${place}的${obj}：${tClause}${meaning}`;
  if (temporal) return `${core}。${temporal}`;
  return `${core}。`;
}

/**
 * One direct equation for a spec: y_t = …
 * @returns {{ tex: string, title: string, description: string } | null}
 */
function buildSpecDirectFormula(spec) {
  if (!spec) return null;
  let x = sourceCallTex(spec);
  const { tensorOps, temporalOps } = splitPipelineOps(spec.transforms || []);
  for (const op of tensorOps) x = applyTransformTex(op, x);
  const scalar = applyReductionTex(spec.reduction, x);
  let tex;
  if (temporalOps.length) {
    tex = wrapTemporal(temporalOps[0], scalar);
    for (let i = 1; i < temporalOps.length; i += 1) {
      // Chain: treat previous y_t as the new scalar stream (rare in this dataset).
      tex = wrapTemporal(temporalOps[i], `y^{(\\mathrm{prev})}_t`);
    }
  } else {
    tex = `y_t=${scalar}`;
  }
  const title = spec.label || `${spec.source_kind} · ${spec.reduction}`;
  return { tex, title, description: buildSpecPlainDescription(spec) };
}

function referenceAnchorForSpec(spec) {
  if (!spec?.id) return "reference.html";
  return `reference.html#obs-${encodeURIComponent(spec.id)}`;
}

/** @deprecated Prefer buildSpecDirectFormula — kept for compatibility. */
function buildSpecDefinitionLatex(spec) {
  const direct = buildSpecDirectFormula(spec);
  if (!direct) return null;
  return [{ kind: "direct", tex: direct.tex, name: direct.title }];
}
