/**
 * Exact formulas and plain-language definitions for ObservableSpec records.
 *
 * The implementation of record is observable_lib.py:
 *   TensorSource -> Transform* -> Reduction -> TemporalOperator*
 *
 * Formulas below intentionally mirror implementation details such as PyTorch's
 * sample-standard-deviation correction and the numerical floors used by the
 * entropy reductions.
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

const SAMPLE_AXIS_SOURCES = new Set([
  "activation",
  "preactivation",
  "logits",
  "attention",
  "gelu_activation",
]);

const FEATURE_AXIS_SOURCES = new Set([
  "activation",
  "preactivation",
  "gelu_activation",
]);

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

function formatTemporalEntry(entry) {
  if (typeof entry === "string") return entry;
  if (Array.isArray(entry)) {
    const [name, params] = entry;
    const args = Object.entries(params || {})
      .map(([key, value]) => `${key}=${value}`)
      .join(",");
    return `${name}(${args})`;
  }
  if (entry && typeof entry === "object") {
    const name = entry.name || entry.op || "";
    const params = entry.params || {};
    const args = Object.entries(params)
      .map(([key, value]) => `${key}=${value}`)
      .join(",");
    return `${name}(${args})`;
  }
  return "";
}

function temporalOpsFromCanonicalId(spec) {
  const id = String(spec?.id || spec?.canonical_id || "");
  const parts = id.split("::");
  if (parts.length < 5 || !parts[4] || parts[4] === "-") return [];
  return parts[4].split("|").filter(Boolean);
}

/**
 * Accept both the corrected manifest schema (`temporal`) and historical
 * manifests, where temporal operators survive only in the canonical id.
 */
function pipelineOpsForSpec(spec) {
  const tensorOps = [];
  const legacyTemporalOps = [];
  for (const raw of spec?.transforms || []) {
    const name = parseOpName(raw);
    if (TEMPORAL_NAMES[name] && name !== "identity") legacyTemporalOps.push(raw);
    else if (name !== "identity") tensorOps.push(raw);
  }

  const explicitTemporal = (spec?.temporal || [])
    .map(formatTemporalEntry)
    .filter(Boolean);
  const temporalOps = explicitTemporal.length
    ? explicitTemporal
    : temporalOpsFromCanonicalId(spec);
  return {
    tensorOps,
    temporalOps: temporalOps.length ? temporalOps : legacyTemporalOps,
  };
}

function escapeTexPath(sel) {
  return String(sel || "")
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([#%&{}])/g, "\\$1")
    .replace(/_/g, "\\_");
}

/** Source tensor at observation index t. */
function sourceCallTex(spec) {
  const path = escapeTexPath(spec.selector || spec.ui_module || "?");
  const kind = spec.source_kind || "tensor";
  const map = {
    weight: "\\Theta",
    grad: "G",
    update: "U",
    opt_m: "M",
    opt_v: "V",
    activation: "A",
    preactivation: "P",
    logits: "Z",
    attention: "\\mathcal{A}",
    gelu_activation: "H",
  };
  const symbol = map[kind] || "X";
  return `${symbol}_{t}^{[\\mathtt{${path}}]}`;
}

function applyTransformTex(op, x, spec) {
  const name = parseOpName(op);
  if (name === "center") {
    if (SAMPLE_AXIS_SOURCES.has(spec?.source_kind)) {
      return `\\big(${x}-\\mathbb{E}_{b}[${x}]\\big)`;
    }
    return `\\big(${x}-\\overline{${x}}\\big)`;
  }
  if (name === "abs") return `\\lvert ${x}\\rvert`;
  if (name === "normalize") {
    const norm = FEATURE_AXIS_SOURCES.has(spec?.source_kind)
      ? `\\|${x}\\|_{2,f}`
      : `\\|${x}\\|_2`;
    return `\\dfrac{${x}}{\\max(${norm},10^{-12})}`;
  }
  if (name === "square") return `(${x})^{2}`;
  return x;
}

/**
 * Reduce tensor x to a scalar. Unless axes are shown explicitly, x_i denotes
 * the N flattened elements. PyTorch torch.std uses correction=1 by default.
 */
function applyReductionTex(reduction, x) {
  switch (reduction) {
    case "mean":
      return `\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i`;
    case "std":
      return (
        `\\sqrt{\\dfrac{1}{N-1}\\sum_{i=1}^{N}` +
        `\\big((${x})_i-\\mu\\big)^2},\\quad` +
        `\\mu=\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i`
      );
    case "min":
      return `\\min_i(${x})_i`;
    case "max":
      return `\\max_i(${x})_i`;
    case "l1_norm":
      return `\\sum_{i=1}^{N}\\lvert(${x})_i\\rvert`;
    case "l2_norm":
      return `\\sqrt{\\sum_{i=1}^{N}(${x})_i^2}`;
    case "rms":
      return `\\sqrt{\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i^2}`;
    case "max_abs":
      return `\\max_i\\lvert(${x})_i\\rvert`;
    case "sparsity":
      return (
        `\\dfrac{1}{N}\\sum_{i=1}^{N}` +
        `\\mathbf{1}\\!\\left[\\lvert(${x})_i\\rvert<10^{-6}\\right]`
      );
    case "positive_fraction":
      return `\\dfrac{1}{N}\\sum_{i=1}^{N}\\mathbf{1}[(${x})_i>0]`;
    case "entropy":
      return (
        `-\\sum_i\\widetilde p_i\\log\\widetilde p_i,\\quad` +
        `\\widetilde p_i=\\max\\!\\left(` +
        `\\dfrac{\\lvert(${x})_i\\rvert}{\\sum_j\\lvert(${x})_j\\rvert},10^{-12}\\right),\\quad` +
        `\\sum_j\\lvert(${x})_j\\rvert>10^{-12}`
      );
    case "spectral_norm":
    case "top_singular_value":
      return `\\sigma_{\\max}(${x})`;
    case "trace":
      return `\\sum_i(${x})_{ii}`;
    case "row_std_mean":
      return (
        `\\dfrac{1}{r}\\sum_{i=1}^{r}` +
        `\\sqrt{\\dfrac{1}{c-1}\\sum_{j=1}^{c}` +
        `\\big((${x})_{ij}-\\overline{x}_{i,:}\\big)^2}`
      );
    case "col_std_mean":
      return (
        `\\dfrac{1}{c}\\sum_{j=1}^{c}` +
        `\\sqrt{\\dfrac{1}{r-1}\\sum_{i=1}^{r}` +
        `\\big((${x})_{ij}-\\overline{x}_{:,j}\\big)^2}`
      );
    case "effective_rank":
      return (
        `\\exp\\!\\left(-\\sum_{k\\in K}p_k\\log p_k\\right),\\quad` +
        `M=\\operatorname{mat}(${x})-\\mathbf{1}\\,\\overline m^{\\!\\top},\\quad` +
        `K=\\{k:\\sigma_k(M)^2>10^{-12}\\},\\quad` +
        `p_k=\\dfrac{\\sigma_k(M)^2}{\\sum_{j\\in K}\\sigma_j(M)^2}`
      );
    case "attention_entropy_mean":
      return (
        `\\mathbb{E}_{b,h,q}\\!\\left[-\\sum_{k=0}^{T-1}` +
        `\\widetilde{${x}}_{b,h,q,k}\\log\\widetilde{${x}}_{b,h,q,k}\\right],\\quad` +
        `\\widetilde{${x}}=\\max(${x},10^{-12})`
      );
    case "attention_entropy_min":
      return (
        `\\min_{b,h,q}\\!\\left[-\\sum_{k=0}^{T-1}` +
        `\\widetilde{${x}}_{b,h,q,k}\\log\\widetilde{${x}}_{b,h,q,k}\\right]` +
        `=-(T-1)\\varepsilon\\log\\varepsilon,\\quad` +
        `\\widetilde{${x}}=\\max(${x},\\varepsilon),\\quad` +
        `\\varepsilon=10^{-12}\\quad(q=0\\text{ guarantees the minimum})`
      );
    case "attention_sink_first_token":
      return `\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}[(${x})_{b,h,q,0}]`;
    case "attention_sink_domination":
      return `\\max_{0\\le k<T}\\mathbb{E}_{b,h,\\,q=0,\\ldots,T-1}[(${x})_{b,h,q,k}]`;
    case "attention_sink_ratio":
      return `T\\,\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}[(${x})_{b,h,q,0}]`;
    case "activation_rate":
      return `\\dfrac{1}{N}\\sum_{i=1}^{N}\\mathbf{1}[(${x})_i>0]`;
    case "massive_activation_peak_ratio":
      return (
        `\\dfrac{\\max_i\\lvert(${x})_i\\rvert}` +
        `{\\sqrt{N^{-1}\\sum_i(${x})_i^2}}`
      );
    case "massive_activation_outlier_fraction":
      return (
        `\\dfrac{1}{N}\\sum_i\\mathbf{1}\\!\\left[` +
        `\\lvert(${x})_i\\rvert>3\\sqrt{N^{-1}\\sum_j(${x})_j^2}\\right]`
      );
    case "massive_neuron_fraction":
      return (
        `\\dfrac{1}{F}\\sum_{f=1}^{F}\\mathbf{1}\\!\\left[` +
        `\\max_{b,q}\\lvert(${x})_{b,q,f}\\rvert>` +
        `3\\sqrt{(BTF)^{-1}\\sum_{b,q,j}(${x})_{b,q,j}^{2}}\\right]`
      );
    default: {
      const safe = String(reduction || "R").replace(/[^a-zA-Z0-9_]/g, "");
      return `\\mathcal{R}_{\\mathrm{${safe}}}(${x})`;
    }
  }
}

function temporalStageTex(raw, inputName, outputName) {
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "identity") return `${outputName}_t=${inputName}_t`;
  if (name === "delta") {
    return `${outputName}_t=${inputName}_t-${inputName}_{t-1}`;
  }
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    return `${outputName}_t=${a}\\,${outputName}_{t-1}+(1-${a})${inputName}_t`;
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return (
      `${outputName}_t=` +
      `\\dfrac{\\sum_{j=0}^{n-1}(j-\\overline j)` +
      `(${inputName}_{t-n+1+j}-\\overline{${inputName}})}` +
      `{\\sum_{j=0}^{n-1}(j-\\overline j)^2},\\quad 2\\le n\\le${w}`
    );
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return (
      `${outputName}_t=` +
      `\\sqrt{\\dfrac{1}{n}\\sum_{j=0}^{n-1}` +
      `(${inputName}_{t-j}-\\overline{${inputName}})^2},\\quad 2\\le n\\le${w}`
    );
  }
  if (name === "curvature") {
    return `${outputName}_t=${inputName}_t-2${inputName}_{t-1}+${inputName}_{t-2}`;
  }
  return `${outputName}_t=${inputName}_t`;
}

function buildTemporalFormula(scalarTex, temporalOps) {
  if (!temporalOps.length) return `y_t=${scalarTex}`;
  const lines = [`z_t=${scalarTex}`];
  let inputName = "z";
  temporalOps.forEach((raw, index) => {
    const outputName = index === temporalOps.length - 1 ? "y" : `z^{(${index + 1})}`;
    lines.push(temporalStageTex(raw, inputName, outputName));
    inputName = outputName;
  });
  return `\\begin{aligned}${lines.join("\\\\[2pt]")}\\end{aligned}`;
}

function modulePlaceEn(spec) {
  const role = spec.role || "";
  const sel = spec.selector || "";
  const layer = spec.layer;
  const L = layer === null || layer === undefined ? null : Number(layer);
  const layerPrefix = L == null ? "" : `layer ${L} `;

  if (role === "wte" || sel.includes("wte")) {
    return "the tied token-embedding / LM-head matrix (wte)";
  }
  if (role === "wpe" || sel.includes("wpe")) return "the position-embedding matrix (wpe)";
  if (role === "ln_f" || sel.includes("ln_f")) return "the final LayerNorm (ln_f)";
  if (role === "lm_head" || sel.includes("lm_head")) return "the LM head";
  if (role === "ln_1" || /\.ln_1/.test(sel)) {
    return `${layerPrefix}pre-attention LayerNorm (ln_1)`;
  }
  if (role === "ln_2" || /\.ln_2/.test(sel)) {
    return `${layerPrefix}pre-MLP LayerNorm (ln_2)`;
  }
  if (role === "attn.c_attn" || sel.includes("c_attn")) {
    return `${layerPrefix}attention QKV projection (c_attn)`;
  }
  if (role === "attn.c_proj" || /attn\.c_proj/.test(sel)) {
    return `${layerPrefix}attention output projection (c_proj)`;
  }
  if (role === "attn" || spec.source_kind === "attention") {
    return `${layerPrefix}causal self-attention`;
  }
  if (role === "mlp.c_fc" || sel.includes("c_fc")) {
    return `${layerPrefix}MLP up-projection (c_fc)`;
  }
  if (role === "mlp.c_proj" || /mlp\.c_proj/.test(sel)) {
    return `${layerPrefix}MLP down-projection (c_proj)`;
  }
  if (
    role === "mlp.gelu" ||
    (spec.ui_module || "").includes("gelu") ||
    spec.source_kind === "gelu_activation"
  ) {
    return `${layerPrefix}MLP GELU`;
  }
  return spec.ui_module ? `${layerPrefix}${spec.ui_module}` : sel || "this module";
}

function activationObjectEn(spec, isInput) {
  const role = spec.role || "";
  if (isInput) {
    if (role === "mlp.c_fc") {
      return "input passed to the up-projection (the ln_2 output), before the affine map";
    }
    if (role === "attn.c_attn") {
      return "input passed to the QKV projection (the ln_1 output), before the affine map";
    }
    if (role === "attn.c_proj") {
      return "concatenated attention-head output passed into c_proj, before the affine map";
    }
    return "input tensor passed to this Linear, before its affine map";
  }
  if (role === "mlp.c_fc") return "affine up-projection output, before GELU";
  if (role === "mlp.c_proj") return "affine down-projection output, before dropout";
  if (role === "attn.c_attn") return "concatenated Q/K/V affine-projection output";
  if (role === "attn.c_proj") return "attention affine-projection output, before residual dropout";
  return "output tensor of this Linear";
}

function sourceObjectEn(spec) {
  const role = spec.role || "";
  switch (spec.source_kind) {
    case "weight":
      if (role.startsWith("ln_") || role === "ln_f") {
        return "LayerNorm scale parameter vector Θ_t";
      }
      if (role === "wte") {
        return "tied vocabulary×embedding parameter matrix Θ_t";
      }
      if (role === "wpe") {
        return "position×embedding parameter matrix Θ_t";
      }
      return "Linear parameter matrix Θ_t (PyTorch layout: out × in)";
    case "grad":
      return "post-backward parameter gradient G_t=∂ℒ_t/∂Θ_t";
    case "update":
      return (
        "accumulated parameter change U_t=Θ_t−Θ_{t^-}, where t^- is the previous " +
        "update-observation instant (possibly several optimizer steps earlier)"
      );
    case "opt_m":
      return "Adam first-moment state M_t (exp_avg; not bias-corrected)";
    case "opt_v":
      return "Adam second-moment state V_t (exp_avg_sq; not bias-corrected)";
    case "activation":
      return activationObjectEn(spec, false);
    case "preactivation":
      return activationObjectEn(spec, true);
    case "logits":
      return "raw LM-head logits Z_t over the vocabulary (before softmax)";
    case "attention":
      return (
        "recomputed pre-dropout causal-softmax probability tensor " +
        "𝒜_t∈ℝ^{B×H×T×T}, with future-key entries zero"
      );
    case "gelu_activation":
      return "post-GELU MLP hidden tensor H_t∈ℝ^{B×T×F}";
    default:
      return spec.source_kind || "source tensor";
  }
}

function transformClauseEn(tensorOps, spec) {
  if (!tensorOps.length) return "";
  const parts = tensorOps.map((op) => {
    const name = parseOpName(op);
    if (name === "center") {
      return SAMPLE_AXIS_SOURCES.has(spec?.source_kind)
        ? "subtracts the batch-axis mean independently at every remaining index"
        : "subtracts the global element mean";
    }
    if (name === "abs") return "takes elementwise absolute values";
    if (name === "normalize") {
      return FEATURE_AXIS_SOURCES.has(spec?.source_kind)
        ? "L2-normalizes each vector along its feature axis, with denominator clamped to 10⁻¹²"
        : "globally L2-normalizes the tensor, with denominator clamped to 10⁻¹²";
    }
    if (name === "square") return "squares every element";
    return `applies transform ${name}`;
  });
  return `${parts.join(", then ")}, then `;
}

function reductionMeaningEn(reduction, sourceKind) {
  switch (reduction) {
    case "mean":
      return "takes the arithmetic mean over all N elements";
    case "std":
      return "takes PyTorch's sample standard deviation over all elements (correction=1, denominator N−1)";
    case "min":
      return "takes the minimum element";
    case "max":
      return "takes the maximum element";
    case "l1_norm":
      return "sums the absolute values (L1 norm)";
    case "l2_norm":
      return "takes the Euclidean/Frobenius magnitude (L2 norm)";
    case "rms":
      return "takes RMS=√[(1/N)Σxᵢ²], a size-normalized typical magnitude";
    case "max_abs":
      return "takes the largest absolute element";
    case "sparsity":
      return "reports the near-zero fraction |x|<10⁻⁶ (threshold sparsity, not exact-zero sparsity)";
    case "positive_fraction":
      return "reports the x>0 fraction (sign balance; for generic tensors this is not a firing-rate definition)";
    case "entropy": {
      const caveat = sourceKind === "logits"
        ? " This is magnitude entropy of raw logits, not predictive entropy of softmax(logits)."
        : "";
      return (
        "takes natural-log Shannon entropy of flattened |x| mass, using the implementation's " +
        "10⁻¹² probability floor (returns NaN when total |x|≤10⁻¹²)." + caveat
      );
    }
    case "spectral_norm":
    case "top_singular_value":
      return "takes the largest singular value (matrix operator 2-norm)";
    case "trace":
      return "sums the diagonal of a square matrix (returns NaN for a non-square tensor)";
    case "row_std_mean":
      return "averages correction=1 standard deviations within output rows";
    case "col_std_mean":
      return "averages correction=1 standard deviations within input columns";
    case "effective_rank":
      return (
        "flattens non-feature axes into samples, centers each feature column, and exponentiates " +
        "the entropy of normalized squared singular values above 10⁻¹²"
      );
    case "attention_entropy_mean":
      return (
        "averages natural-log key-distribution entropy over every batch, head, and query, " +
        "including q=0 and using a 10⁻¹² floor"
      );
    case "attention_entropy_min":
      return (
        "takes the minimum key-distribution entropy over every batch, head, and query. " +
        "Because q=0 can attend only to k=0, this metric is structurally pinned near zero and " +
        "has little diagnostic value in a causal model"
      );
    case "attention_sink_first_token":
      return "averages the attention probability assigned to key 0 by non-first queries q=1,…,T−1";
    case "attention_sink_domination":
      return (
        "finds the key position with the largest attention probability averaged over batch, " +
        "head, and all queries. Causally unavailable query-key pairs enter as zero, so earlier " +
        "keys have a structural exposure advantage"
      );
    case "attention_sink_ratio":
      return (
        "multiplies first-key mass by T, i.e. compares it with the full-length reference 1/T. " +
        "This is not a causal-uniform normalization: even uniform attention over each query's " +
        "available prefix generally gives a value greater than 1"
      );
    case "activation_rate":
      return "reports the fraction of post-GELU elements that are strictly positive";
    case "massive_activation_peak_ratio":
      return "divides the largest absolute activation by the global RMS (returns NaN when RMS≤10⁻¹²)";
    case "massive_activation_outlier_fraction":
      return "reports the element fraction with |x|>3·global-RMS (returns NaN when RMS≤10⁻¹²)";
    case "massive_neuron_fraction":
      return (
        "reports the feature fraction whose maximum |x| over batch×token exceeds 3·global-RMS; " +
        "this is a peak criterion, not a persistence criterion"
      );
    default:
      return `applies scalar reduction "${reduction}"`;
  }
}

function temporalMeaningEn(raw) {
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "identity") return "passes the scalar stream through unchanged";
  if (name === "delta") {
    return "takes the difference from the previous recorded observation (the first difference is undefined)";
  }
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    return `applies EMA with α=${a}, initialized by the first finite input`;
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return (
      `fits an OLS slope against observation index over up to the latest ${w} finite values ` +
      "(not against raw training-step distance)"
    );
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `takes population standard deviation over up to the latest ${w} finite observations`;
  }
  if (name === "curvature") return "takes the three-point second difference";
  return `applies temporal operator ${name}`;
}

function isDegenerateCenteredMean(spec, tensorOps) {
  return (
    spec?.reduction === "mean" &&
    tensorOps.some((op) => parseOpName(op) === "center") &&
    SAMPLE_AXIS_SOURCES.has(spec?.source_kind)
  );
}

function buildSpecPlainDescription(spec) {
  if (!spec) return "";
  const place = modulePlaceEn(spec);
  const source = sourceObjectEn(spec);
  const { tensorOps, temporalOps } = pipelineOpsForSpec(spec);
  const transform = transformClauseEn(tensorOps, spec);
  const reduction = reductionMeaningEn(spec.reduction, spec.source_kind);

  let text = `At ${place}, reads the ${source}; ${transform}${reduction}.`;
  if (temporalOps.length) {
    text += ` On the resulting scalar stream, it ${temporalOps
      .map(temporalMeaningEn)
      .join(", then ")}.`;
  }
  if (isDegenerateCenteredMean(spec, tensorOps)) {
    text += (
      " Important: batch-centering makes the subsequent global mean exactly zero in exact " +
      "arithmetic, so this historical observable is structurally degenerate; nonzero values " +
      "are floating-point noise."
    );
  }
  return text;
}

/**
 * Build the complete, implementation-aligned equation for one catalog record.
 */
function buildSpecDirectFormula(spec) {
  if (!spec) return null;
  let x = sourceCallTex(spec);
  const { tensorOps, temporalOps } = pipelineOpsForSpec(spec);
  for (const op of tensorOps) x = applyTransformTex(op, x, spec);
  const scalar = applyReductionTex(spec.reduction, x);
  const tex = buildTemporalFormula(scalar, temporalOps);
  const title = spec.label || `${spec.source_kind} · ${spec.reduction}`;
  return { tex, title, description: buildSpecPlainDescription(spec) };
}

function referenceAnchorForSpec(spec) {
  if (!spec?.id) return "reference.html";
  return `reference.html#obs-${encodeURIComponent(spec.id)}`;
}

/** @deprecated Prefer buildSpecDirectFormula; retained for compatibility. */
function buildSpecDefinitionLatex(spec) {
  const direct = buildSpecDirectFormula(spec);
  if (!direct) return null;
  return [{ kind: "direct", tex: direct.tex, name: direct.title }];
}
