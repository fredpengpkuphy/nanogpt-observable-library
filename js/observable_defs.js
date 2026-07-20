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

function modulePlaceEn(spec) {
  const role = spec.role || "";
  const sel = spec.selector || "";
  const layer = spec.layer;
  const L = layer === null || layer === undefined ? null : Number(layer);

  const layerPrefix = L == null ? "" : `layer ${L} `;

  if (role === "wte" || sel.includes("wte")) return "token embedding (wte)";
  if (role === "wpe" || sel.includes("wpe")) return "position embedding (wpe)";
  if (role === "ln_f" || sel.includes("ln_f")) return "final LayerNorm (ln_f)";
  if (role === "lm_head" || sel.includes("lm_head")) return "LM head (vocab logits projection)";

  if (role === "ln_1" || /\.ln_1/.test(sel)) return `${layerPrefix}pre-attention LayerNorm (ln_1)`;
  if (role === "ln_2" || /\.ln_2/.test(sel)) return `${layerPrefix}pre-MLP LayerNorm (ln_2)`;
  if (role === "attn.c_attn" || sel.includes("c_attn")) {
    return `${layerPrefix}attention QKV projection (c_attn)`;
  }
  if (role === "attn.c_proj" || /attn\.c_proj/.test(sel)) {
    return `${layerPrefix}attention output projection (c_proj)`;
  }
  if (
    role === "attn" ||
    /^h\.\d+\.attn$/.test(spec.ui_module || "") ||
    (sourceLooksLikeAttentionWeights(spec) && !sel.includes("c_attn") && !sel.includes("c_proj"))
  ) {
    return `${layerPrefix}causal self-attention (softmax attention weights)`;
  }
  if (role === "mlp.c_fc" || sel.includes("c_fc")) {
    return `${layerPrefix}MLP up-projection (c_fc)`;
  }
  if (role === "mlp.c_proj" || /mlp\.c_proj/.test(sel)) {
    return `${layerPrefix}MLP down-projection (c_proj)`;
  }
  if (role === "mlp.gelu" || (spec.ui_module || "").includes("gelu") || spec.source_kind === "gelu_activation") {
    return `${layerPrefix}MLP post-GELU hidden state`;
  }

  if (spec.ui_module) return `${layerPrefix}${spec.ui_module}`;
  return sel || "this module";
}

function sourceLooksLikeAttentionWeights(spec) {
  return spec.source_kind === "attention" || String(spec.reduction || "").startsWith("attention_");
}

function sourceObjectEn(spec) {
  switch (spec.source_kind) {
    case "weight":
      return "parameter weight tensor θ";
    case "grad":
      return "backpropagated gradient g=∇_θℒ";
    case "update":
      return "parameter update between adjacent observation steps u=θ_t−θ_{t−1}";
    case "opt_m":
      return "Adam first-moment estimate m̂ (exp_avg)";
    case "opt_v":
      return "Adam second-moment estimate v̂ (exp_avg_sq)";
    case "activation":
      return "forward activation a of this Linear";
    case "preactivation":
      return "pre-activation input ã of this Linear";
    case "logits":
      return "LM-head logits z over the vocabulary";
    case "attention":
      return "causal softmax attention matrix A∈ℝ^{B×H×T×T} (query→key probabilities)";
    case "gelu_activation":
      return "post-GELU hidden activations in the MLP";
    default:
      return spec.source_kind || "tensor";
  }
}

function transformClauseEn(tensorOps) {
  if (!tensorOps?.length) return "";
  const parts = [];
  for (const op of tensorOps) {
    const name = parseOpName(op);
    if (name === "center") {
      parts.push("first subtract the mean along the batch (sample) axis, or over all elements if there is no batch axis");
    } else if (name === "abs") {
      parts.push("first take absolute values");
    } else if (name === "normalize") {
      parts.push("first L2-normalize along the feature axis (or globally if there is no feature axis)");
    } else if (name === "square") {
      parts.push("first square the elements");
    } else {
      parts.push(`first apply transform ${name}`);
    }
  }
  return parts.length ? parts.join("; ") + "; then " : "";
}

function reductionMeaningEn(reduction, sourceKind) {
  const isAttn = sourceKind === "attention" || String(reduction).startsWith("attention_");
  const isGelu = sourceKind === "gelu_activation" || String(reduction).startsWith("massive_") || reduction === "activation_rate";

  switch (reduction) {
    case "mean":
      return "mean over all elements (overall bias / average level)";
    case "std":
      return "standard deviation over all elements (spread / scale fluctuation)";
    case "min":
      return "minimum over all elements";
    case "max":
      return "maximum over all elements";
    case "l1_norm":
      return "L1 norm (sum of absolute values), a total-mass / sparsity-related scale";
    case "l2_norm":
      return "L2 norm (Euclidean length), the overall energy / scale of the tensor";
    case "rms":
      return "RMS = √E[x²], a typical magnitude (sign-insensitive)";
    case "max_abs":
      return "maximum absolute value — the most extreme single-element magnitude";
    case "sparsity":
      return "fraction of near-zero elements (|x|<10⁻⁶)";
    case "positive_fraction":
      return "fraction of positive elements (sign bias / “active” share)";
    case "entropy":
      return "Shannon entropy after normalizing |x| into a probability mass: higher = more diffuse, lower = concentrated on few components";
    case "spectral_norm":
    case "top_singular_value":
      return "largest singular value σ_max of the weight matrix (operator norm / max gain)";
    case "trace":
      return "trace tr(W) of a square matrix (sum of diagonal entries; undefined for non-square)";
    case "row_std_mean":
      return "mean of per-output-row standard deviations (row-wise heterogeneity)";
    case "col_std_mean":
      return "mean of per-input-column standard deviations (column-wise heterogeneity)";
    case "effective_rank":
      return "effective rank from the entropy of centered covariance eigenvalues: higher = energy spread over more directions, lower = low-rank / collapse";
    case "attention_entropy_mean":
      return isAttn
        ? "mean Shannon entropy of each (batch, head, query) key distribution: higher = more diffuse attention, lower = sharper focus"
        : "mean attention entropy";
    case "attention_entropy_min":
      return "minimum attention entropy over all (batch, head, query): smaller = extremely peaked attention exists";
    case "attention_sink_first_token":
      return "mean attention mass that non-first-token queries assign to key=0 (sequence start) — attention-sink strength";
    case "attention_sink_domination":
      return "max average attention mass received by any key position: near 1 means almost all mass sinks to one key";
    case "attention_sink_ratio":
      return "first-token sink mass relative to uniform attention 1/T (= T·E[A_{q,0}]): >1 means the first position is over-attended";
    case "activation_rate":
      return isGelu
        ? "fraction of post-GELU hidden elements with x>0 (firing rate)"
        : "fraction of elements with x>0 (firing rate)";
    case "massive_activation_peak_ratio":
      return "max|x| / RMS(x): larger means a strongly amplified massive-activation spike vs background";
    case "massive_activation_outlier_fraction":
      return "fraction of elements with |x|>3·RMS(x) (elementwise outlier / massive-activation coverage)";
    case "massive_neuron_fraction":
      return "fraction of hidden neurons whose peak |x| over all batch×time exceeds 3·RMS(global) — massive-neuron share";
    default:
      return `scalar summary “${reduction}”`;
  }
}

function temporalClauseEn(temporalOps) {
  if (!temporalOps?.length) return "";
  const raw = temporalOps[0];
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "delta") return " Then take a first difference along the training-step series (change vs the previous observation).";
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    return ` Then apply EMA smoothing along the training-step series (α=${a}).`;
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return ` Then fit an OLS slope over the last ${w} observations (short-term trend).`;
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return ` Then take the rolling standard deviation over the last ${w} observations (short-term volatility).`;
  }
  if (name === "curvature") return " Then take a three-point second difference (acceleration / curvature).";
  return "";
}

/**
 * Direct plain-language description: what this observable is measuring.
 */
function buildSpecPlainDescription(spec) {
  if (!spec) return "";
  const place = modulePlaceEn(spec);
  const obj = sourceObjectEn(spec);
  const { tensorOps, temporalOps } = splitPipelineOps(spec.transforms || []);
  const tClause = transformClauseEn(tensorOps);
  const meaning = reductionMeaningEn(spec.reduction, spec.source_kind);
  const temporal = temporalClauseEn(temporalOps);

  const core = `Measures the ${obj} of ${place}: ${tClause}${meaning}`;
  if (temporal) return `${core}.${temporal}`;
  return `${core}.`;
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
