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
    return "the token embedding / LM head (wte)";
  }
  if (role === "wpe" || sel.includes("wpe")) return "the position embedding (wpe)";
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
    if (role === "mlp.c_fc") return "the input";
    if (role === "attn.c_attn") return "the input";
    if (role === "attn.c_proj") return "the combined attention-head output";
    return "the module input";
  }
  if (role === "mlp.c_fc") return "the output before GELU";
  if (role === "mlp.c_proj") return "the MLP down-projection output";
  if (role === "attn.c_attn") return "the combined Q/K/V projection output";
  if (role === "attn.c_proj") return "the attention output projection";
  return "the module output";
}

function sourceObjectEn(spec) {
  const role = spec.role || "";
  switch (spec.source_kind) {
    case "weight":
      if (role.startsWith("ln_") || role === "ln_f") {
        return "the LayerNorm scale weights";
      }
      if (role === "wte") {
        return "the shared weights";
      }
      if (role === "wpe") {
        return "the weights";
      }
      return "the parameter weights";
    case "grad":
      return "the training gradients";
    case "update":
      return "the weight change since the previous observation";
    case "opt_m":
      return "Adam's running gradient average";
    case "opt_v":
      return "Adam's running squared-gradient average";
    case "activation":
      return activationObjectEn(spec, false);
    case "preactivation":
      return activationObjectEn(spec, true);
    case "logits":
      return "the raw vocabulary scores before softmax";
    case "attention":
      return "the attention probabilities";
    case "gelu_activation":
      return "the activations";
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
        ? "subtracts the batch average"
        : "subtracts the overall average";
    }
    if (name === "abs") return "takes absolute values";
    if (name === "normalize") {
      return FEATURE_AXIS_SOURCES.has(spec?.source_kind)
        ? "normalizes each feature vector"
        : "normalizes the whole tensor";
    }
    if (name === "square") return "squares the values";
    return `applies transform ${name}`;
  });
  return `${parts.join(", then ")}, then `;
}

function reductionMeaningEn(reduction, sourceKind) {
  switch (reduction) {
    case "mean":
      return "takes the average value";
    case "std":
      return "measures how widely the values are spread";
    case "min":
      return "finds the smallest value";
    case "max":
      return "finds the largest value";
    case "l1_norm":
      return "adds the absolute values to measure total magnitude";
    case "l2_norm":
      return "measures the overall magnitude";
    case "rms":
      return "measures the typical magnitude";
    case "max_abs":
      return "finds the most extreme magnitude";
    case "sparsity":
      return "reports the fraction of values that are nearly zero";
    case "positive_fraction":
      return "reports the fraction of positive values";
    case "entropy": {
      const caveat = sourceKind === "logits"
        ? " For logits, this describes score magnitudes rather than prediction uncertainty."
        : "";
      return (
        "measures how evenly the total magnitude is distributed; higher means more diffuse" +
        caveat
      );
    }
    case "spectral_norm":
    case "top_singular_value":
      return "measures the strongest amplification of the weight matrix";
    case "trace":
      return "adds the diagonal values";
    case "row_std_mean":
      return "measures the average variation within output rows";
    case "col_std_mean":
      return "measures the average variation within input columns";
    case "effective_rank":
      return "estimates how many independent feature directions are being used";
    case "attention_entropy_mean":
      return "measures how spread out attention is on average; higher means more diffuse attention";
    case "attention_entropy_min":
      return (
        "finds the sharpest attention pattern; it stays near zero because the first token " +
        "attends only to itself, limiting its usefulness"
      );
    case "attention_sink_first_token":
      return "measures how much later tokens attend to the first token";
    case "attention_sink_domination":
      return (
        "measures concentration on one position; earlier positions naturally score higher " +
        "because more tokens can attend to them"
      );
    case "attention_sink_ratio":
      return (
        "compares first-token attention with the uniform level; because attention is causal, " +
        "values above 1 do not by themselves indicate a sink"
      );
    case "activation_rate":
      return "reports the fraction of post-GELU activations that are positive";
    case "massive_activation_peak_ratio":
      return "compares the largest activation with the typical activation size";
    case "massive_activation_outlier_fraction":
      return "reports the fraction of unusually large activations";
    case "massive_neuron_fraction":
      return "reports the fraction of neurons that produce an unusually large activation";
    default:
      return `applies scalar reduction "${reduction}"`;
  }
}

function temporalMeaningEn(raw) {
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "identity") return "keeps the value unchanged";
  if (name === "delta") return "measures the change from the previous observation";
  if (name === "ema") return "smooths the changes with an exponential moving average";
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `measures the recent trend over the latest ${w} observations`;
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return `measures short-term variability over the latest ${w} observations`;
  }
  if (name === "curvature") return "measures how quickly the trend is changing";
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

  if (isDegenerateCenteredMean(spec, tensorOps)) {
    return (
      `Tracks ${source} at ${place}. It averages the values after subtracting the batch ` +
      "average, so the result should be zero. Tiny values are numerical noise; the chart " +
      "shows their recent trend."
    );
  }

  let text = `Tracks ${source} at ${place}. It ${transform}${reduction}.`;
  if (temporalOps.length) {
    text += ` Across observations, it ${temporalOps
      .map(temporalMeaningEn)
      .join(", then ")}.`;
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
