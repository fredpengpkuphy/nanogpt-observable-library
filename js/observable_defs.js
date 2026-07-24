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

function observableDisplayLabel(spec, knownTemporalOps = null) {
  const base =
    spec?.label || `${spec?.source_kind || "observable"} · ${spec?.reduction || ""}`;
  const temporalOps = knownTemporalOps || pipelineOpsForSpec(spec).temporalOps;
  if (!temporalOps.length) return base;
  if (temporalOps.every((op) => base.includes(op))) {
    return base.replace(temporalOps.join(">"), temporalOps.join(" → "));
  }
  return `${base} · ${temporalOps.join(" → ")}`;
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
  if (kind === "grad") {
    return `\\nabla_{\\Theta^{[\\mathtt{${path}}]}}\\mathcal{L}_t`;
  }
  if (kind === "update") {
    return (
      `(\\Theta_t-\\Theta_{t^-})^{[\\mathtt{${path}}]}`
    );
  }
  const map = {
    weight: "\\Theta",
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
      return `\\operatorname{center}_{b}(${x})=${x}-\\mathbb{E}_{b}[${x}]`;
    }
    return `\\operatorname{center}(${x})=${x}-\\overline{${x}}`;
  }
  if (name === "abs") return `\\lvert ${x}\\rvert`;
  if (name === "normalize") {
    const norm = FEATURE_AXIS_SOURCES.has(spec?.source_kind)
      ? `\\|${x}\\|_{2,f}`
      : `\\|${x}\\|_2`;
    return `\\operatorname{normalize}(${x})=\\dfrac{${x}}{\\max(${norm},10^{-12})}`;
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
      return `\\operatorname{mean}(${x})=\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i`;
    case "std":
      return {
        main:
          `\\operatorname{std}(${x})=` +
          `\\sqrt{\\dfrac{1}{N-1}\\sum_{i=1}^{N}\\big((${x})_i-\\mu\\big)^2}`,
        details: [`\\mu=\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i`],
      };
    case "min":
      return `\\operatorname{min}(${x})=\\min_i(${x})_i`;
    case "max":
      return `\\operatorname{max}(${x})=\\max_i(${x})_i`;
    case "l1_norm":
      return `\\|${x}\\|_1=\\sum_{i=1}^{N}\\lvert(${x})_i\\rvert`;
    case "l2_norm":
      return `\\|${x}\\|_2=\\sqrt{\\sum_{i=1}^{N}(${x})_i^2}`;
    case "rms":
      return `\\operatorname{RMS}(${x})=\\sqrt{\\dfrac{1}{N}\\sum_{i=1}^{N}(${x})_i^2}`;
    case "max_abs":
      return `\\operatorname{maxabs}(${x})=\\max_i\\lvert(${x})_i\\rvert`;
    case "sparsity":
      return (
        `\\operatorname{nearzero}(${x})=\\dfrac{1}{N}\\sum_{i=1}^{N}` +
        `\\mathbf{1}\\!\\left[\\lvert(${x})_i\\rvert<10^{-6}\\right]`
      );
    case "positive_fraction":
      return (
        `\\operatorname{positive\\,frac}(${x})=` +
        `\\dfrac{1}{N}\\sum_{i=1}^{N}\\mathbf{1}[(${x})_i>0]`
      );
    case "entropy":
      return {
        main: `\\operatorname{entropy}(${x})=-\\sum_i\\widetilde p_i\\log\\widetilde p_i`,
        details: [
          `\\widetilde p_i=\\max\\!\\left(` +
            `\\dfrac{\\lvert(${x})_i\\rvert}{\\sum_j\\lvert(${x})_j\\rvert},10^{-12}\\right)`,
          `\\text{defined when }\\sum_j\\lvert(${x})_j\\rvert>10^{-12}`,
        ],
      };
    case "spectral_norm":
    case "top_singular_value":
      return `\\|${x}\\|_{2}=\\sigma_{\\max}(${x})`;
    case "trace":
      return `\\operatorname{tr}(${x})=\\sum_i(${x})_{ii}`;
    case "row_std_mean":
      return {
        main:
          `\\operatorname{row\\,std}(${x})=` +
          `\\dfrac{1}{r}\\sum_{i=1}^{r}s_i`,
        details: [
          `s_i=\\sqrt{\\dfrac{1}{c-1}\\sum_{j=1}^{c}` +
            `\\big((${x})_{ij}-\\overline{x}_{i,:}\\big)^2}`,
          `\\overline{x}_{i,:}=\\dfrac{1}{c}\\sum_{j=1}^{c}(${x})_{ij}`,
        ],
      };
    case "col_std_mean":
      return {
        main:
          `\\operatorname{col\\,std}(${x})=` +
          `\\dfrac{1}{c}\\sum_{j=1}^{c}s_j`,
        details: [
          `s_j=\\sqrt{\\dfrac{1}{r-1}\\sum_{i=1}^{r}` +
            `\\big((${x})_{ij}-\\overline{x}_{:,j}\\big)^2}`,
          `\\overline{x}_{:,j}=\\dfrac{1}{r}\\sum_{i=1}^{r}(${x})_{ij}`,
        ],
      };
    case "effective_rank":
      return {
        main:
          `\\operatorname{erank}(${x})=` +
          `\\exp\\!\\left(-\\sum_{k\\in K}p_k\\log p_k\\right)`,
        details: [
          `A=\\operatorname{mat}(${x})\\in\\mathbb{R}^{n\\times F}`,
          `\\overline a=\\dfrac{1}{n}\\sum_{i=1}^{n}A_{i,:},\\quad ` +
            `M=A-\\mathbf{1}\\,\\overline a^{\\!\\top}`,
          `K=\\{k:\\sigma_k(M)^2>10^{-12}\\}`,
          `p_k=\\dfrac{\\sigma_k(M)^2}{\\sum_{j\\in K}\\sigma_j(M)^2}`,
          `\\text{defined when }n\\ge2\\text{ and }K\\ne\\varnothing`,
        ],
      };
    case "attention_entropy_mean":
      return {
        main:
          `\\operatorname{mean\\,attn\\,entropy}(${x})=` +
          `\\mathbb{E}_{b,h,q}[H_{b,h,q}]`,
        details: [
          `H_{b,h,q}=-\\sum_{k=0}^{T-1}` +
            `\\widetilde{${x}}_{b,h,q,k}\\log\\widetilde{${x}}_{b,h,q,k}`,
          `\\widetilde{${x}}=\\max(${x},10^{-12})`,
        ],
      };
    case "attention_entropy_min":
      return {
        main:
          `\\operatorname{min\\,attn\\,entropy}(${x})=` +
          `\\min_{b,h,q}H_{b,h,q}`,
        details: [
          `H_{b,h,q}=-\\sum_{k=0}^{T-1}` +
            `\\widetilde{${x}}_{b,h,q,k}\\log\\widetilde{${x}}_{b,h,q,k}`,
          `\\widetilde{${x}}=\\max(${x},\\varepsilon),\\quad\\varepsilon=10^{-12}`,
          `&0\\le\\operatorname{min\\,attn\\,entropy}(${x})\\le` +
            `-(T-1)\\varepsilon\\log\\varepsilon,\\quad\\text{using }q=0`,
        ],
      };
    case "attention_sink_first_token":
      return {
        main:
          `\\operatorname{first\\,token\\,mass}(${x})=` +
          `\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}[(${x})_{b,h,q,0}]`,
        details: [`\\text{defined when }T\\ge2`],
      };
    case "attention_sink_domination":
      return (
        `\\operatorname{sink\\,domination}(${x})=` +
        `\\max_{0\\le k<T}\\mathbb{E}_{b,h,\\,q=0,\\ldots,T-1}[(${x})_{b,h,q,k}]`
      );
    case "attention_sink_ratio":
      return {
        main:
          `\\operatorname{first\\,token\\,ratio}(${x})=` +
          `T\\,\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}[(${x})_{b,h,q,0}],` +
          `\\quad T\\ge2`,
        details: [
          `\\operatorname{first\\,token\\,ratio}(${x})=` +
            `\\mathbb{E}_{b,h,q}[(${x})_{b,h,q,0}],\\quad T=1`,
        ],
      };
    case "activation_rate":
      return (
        `\\operatorname{activation\\,rate}(${x})=` +
        `\\dfrac{1}{N}\\sum_{i=1}^{N}\\mathbf{1}[(${x})_i>0]`
      );
    case "massive_activation_peak_ratio":
      return {
        main:
          `\\operatorname{peak\\,ratio}(${x})=` +
          `\\dfrac{\\max_i\\lvert(${x})_i\\rvert}{r}`,
        details: [
          `r=\\operatorname{RMS}(${x})=\\sqrt{N^{-1}\\sum_i(${x})_i^2}`,
          `\\text{defined when }r>10^{-12}`,
        ],
      };
    case "massive_activation_outlier_fraction":
      return {
        main:
          `\\operatorname{outlier\\,frac}(${x})=` +
          `\\dfrac{1}{N}\\sum_i\\mathbf{1}\\!\\left[\\lvert(${x})_i\\rvert>3r\\right]`,
        details: [
          `r=\\operatorname{RMS}(${x})=\\sqrt{N^{-1}\\sum_i(${x})_i^2}`,
          `\\text{defined when }r>10^{-12}`,
        ],
      };
    case "massive_neuron_fraction":
      return {
        main:
          `\\operatorname{massive\\,neuron\\,frac}(${x})=` +
          `\\dfrac{1}{F}\\sum_{f=1}^{F}\\mathbf{1}[a_f>3r]`,
        details: [
          `a_f=\\max_{b,\\tau}\\lvert(${x})_{b,\\tau,f}\\rvert`,
          `r=\\operatorname{RMS}(${x})=` +
            `\\sqrt{(BTF)^{-1}\\sum_{b,\\tau,j}(${x})_{b,\\tau,j}^{2}}`,
          `\\text{defined when }r>10^{-12}`,
        ],
      };
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
    const previous = `t^{-}_{${inputName}}`;
    return {
      main:
        `${outputName}_t=\\Delta ${inputName}_t=` +
        `${inputName}_t-${inputName}_{${previous}}`,
      details: [
        `${previous}=\\text{previous observation with finite }${inputName}`,
      ],
    };
  }
  if (name === "ema") {
    const a = parseNamedArg(args, "alpha") || "0.9";
    const previous = `t^{-}_{${inputName}}`;
    return {
      main:
        `${outputName}_t=\\operatorname{EMA}_{${a}}(${inputName})_t=` +
        `${a}\\,${outputName}_{${previous}}+(1-${a})${inputName}_t`,
      details: [
        `${previous}=\\text{previous observation with finite }${inputName}`,
        `${outputName}_{t_0}=${inputName}_{t_0},\\quad ` +
          `t_0=\\text{first finite input}`,
      ],
    };
  }
  if (name === "slope") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return {
      main:
        `${outputName}_t=` +
        `\\operatorname{slope}(u_0,\\ldots,u_{n-1})`,
      details: [
        `\\operatorname{slope}(u_0,\\ldots,u_{n-1})=` +
          `\\dfrac{\\sum_{j=0}^{n-1}(j-\\overline j)(u_j-\\overline u)}{D}`,
        `D=\\sum_{j=0}^{n-1}(j-\\overline j)^2,\\quad 2\\le n\\le${w}`,
        `u_0,\\ldots,u_{n-1}=\\text{latest }n\\text{ finite values of }${inputName}`,
        `\\overline j=\\dfrac{n-1}{2}`,
        `\\overline u=\\dfrac{1}{n}\\sum_{j=0}^{n-1}u_j`,
      ],
    };
  }
  if (name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w") || "5";
    return {
      main:
        `${outputName}_t=` +
        `\\operatorname{rolling\\,std}(u_0,\\ldots,u_{n-1})=` +
        `\\sqrt{\\dfrac{1}{n}\\sum_{j=0}^{n-1}(u_j-\\overline u)^2},` +
        `\\quad 2\\le n\\le${w}`,
      details: [
        `u_0,\\ldots,u_{n-1}=\\text{latest }n\\text{ finite values of }${inputName}`,
        `\\overline u=\\dfrac{1}{n}\\sum_{j=0}^{n-1}u_j`,
      ],
    };
  }
  if (name === "curvature") {
    return {
      main: `${outputName}_t=\\Delta^2${inputName}_t=u_2-2u_1+u_0`,
      details: [
        `u_0,u_1,u_2=\\text{latest three finite values of }${inputName}`,
      ],
    };
  }
  return `${outputName}_t=${inputName}_t`;
}

function alignFormulaLine(line) {
  if (line.includes("&")) return line;
  const eq = line.indexOf("=");
  return eq >= 0 ? `${line.slice(0, eq)}&=${line.slice(eq + 1)}` : `&${line}`;
}

function buildFormulaLines(prefixLines, scalarTex, scalarDetails, temporalOps) {
  const lines = [...prefixLines];
  const scalarName = temporalOps.length ? "s" : "y";
  lines.push(`${scalarName}_t=${scalarTex}`);
  lines.push(...scalarDetails);
  let inputName = scalarName;
  temporalOps.forEach((raw, index) => {
    const outputName = index === temporalOps.length - 1 ? "y" : `z^{(${index + 1})}`;
    const stage = temporalStageTex(raw, inputName, outputName);
    if (typeof stage === "string") {
      lines.push(stage);
    } else {
      lines.push(stage.main, ...(stage.details || []));
    }
    inputName = outputName;
  });
  return `\\begin{aligned}${lines.map(alignFormulaLine).join("\\\\[2pt]")}\\end{aligned}`;
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
      if (sourceKind === "logits") {
        return (
          "measures how evenly absolute score magnitudes are distributed; " +
          "it is not prediction uncertainty"
        );
      }
      return "measures how evenly the total magnitude is distributed; higher means more diffuse";
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
  if (name === "delta") return "measures the change from the previous available observation";
  if (name === "ema") return "smooths the values with an exponential moving average";
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

const NOTATION_SYMBOL_TEX = {
  "Θ_t": "\\Theta_t",
  "∇_Θ L_t": "\\nabla_{\\Theta}\\mathcal{L}_t",
  "Θ_t−Θ_t⁻": "\\Theta_t-\\Theta_{t^-}",
  "𝒜_t": "\\mathcal{A}_t",
  "X_t^(m)": "X_t^{(m)}",
  "μ": "\\mu",
  "p̃_i": "\\widetilde{p}_i",
  "1[·]": "\\mathbf{1}[\\cdot]",
  "σ_k(M)": "\\sigma_k(M)",
  "σ_max": "\\sigma_{\\max}",
  "X̃": "\\widetilde{X}",
  "τ": "\\tau",
  "‖·‖₂,f": "\\lVert\\cdot\\rVert_{2,f}",
  "t⁻": "t^-",
  "j̄, ū": "\\bar{j},\\,\\bar{u}",
};

function notationSymbolTex(symbol) {
  return NOTATION_SYMBOL_TEX[symbol] || String(symbol || "");
}

/**
 * Plain-language glossary for every index or shorthand that appears in the
 * displayed formula. Keep this separate from the equation so notation remains
 * readable on narrow screens and accessible without parsing TeX.
 */
function buildSpecNotation(spec) {
  if (!spec) return [];
  const entries = [];
  const seen = new Set();
  const add = (symbol, meaning, tex = notationSymbolTex(symbol)) => {
    if (!symbol || seen.has(symbol)) return;
    seen.add(symbol);
    entries.push({ symbol, tex, meaning });
  };
  const { tensorOps, temporalOps } = pipelineOpsForSpec(spec);
  const reduction = spec.reduction || "";
  const genericElementReductions = new Set([
    "mean", "std", "min", "max", "l1_norm", "l2_norm", "rms", "max_abs",
    "sparsity", "positive_fraction", "entropy", "activation_rate",
    "massive_activation_peak_ratio", "massive_activation_outlier_fraction",
  ]);

  add("t", "observation index: the checkpoint or training step at which this tensor was recorded");
  add("X_t", "the source tensor recorded at observation t");
  const sourceSymbols = {
    weight: ["Θ_t", "model-parameter tensor at observation t"],
    grad: ["∇_Θ L_t", "gradient of the training loss L with respect to the selected parameters Θ"],
    update: ["Θ_t−Θ_t⁻", "change in the selected parameter tensor since its previous observation"],
    activation: ["A_t", "recorded activation tensor"],
    preactivation: ["P_t", "recorded tensor before the module’s nonlinearity"],
    logits: ["Z_t", "recorded vocabulary-logit tensor"],
    attention: ["𝒜_t", "recorded attention-probability tensor"],
    gelu_activation: ["H_t", "recorded post-GELU activation tensor"],
  };
  if (sourceSymbols[spec.source_kind]) add(...sourceSymbols[spec.source_kind]);
  if (tensorOps.length) {
    add("X_t^(m)", "intermediate tensor after the m-th transform in the displayed pipeline");
  }
  add("y_t", "final scalar observable plotted at observation t");
  if (temporalOps.length) {
    add("s_t", "scalar reduction of the tensor before temporal processing");
  }

  if (genericElementReductions.has(reduction)) {
    add("i", "index of one element after the tensor is flattened");
    add("N", "total number of elements in the flattened tensor");
  }
  if (reduction === "std") {
    add("μ", "mean of the N flattened tensor elements");
  }
  if (reduction === "entropy") {
    add("j", "flattened-element index used in the normalization sum");
    add("p̃_i", "normalized absolute magnitude of flattened element i, floored at 10⁻¹²");
  }
  if (reduction === "trace") {
    add("i", "matrix diagonal index; X_ii is the i-th diagonal entry");
  }
  if (["sparsity", "positive_fraction", "activation_rate",
       "massive_activation_outlier_fraction", "massive_neuron_fraction"].includes(reduction)) {
    add("1[·]", "indicator: 1 when the condition in brackets is true, otherwise 0");
  }
  if (["row_std_mean", "col_std_mean"].includes(reduction)) {
    add("i", "matrix row index");
    add("j", "matrix column index");
    add("r", "number of matrix rows");
    add("c", "number of matrix columns");
  }
  if (reduction === "effective_rank") {
    add("n", "number of samples after reshaping the tensor into a matrix");
    add("F", "feature dimension of the reshaped matrix");
    add("k", "singular-value index");
    add("K", "indices of singular values whose squared magnitude exceeds 10⁻¹²");
    add("σ_k(M)", "k-th singular value of the centered matrix M");
    add("p_k", "fraction of total retained squared singular-value mass carried by component k");
    add("A", "matrix obtained by reshaping the source tensor into n samples by F features");
    add("M", "A after subtracting its mean feature vector from every sample");
  }
  if (["spectral_norm", "top_singular_value"].includes(reduction)) {
    add("σ_max", "largest singular value of the source matrix");
  }

  const attentionReductions = new Set([
    "attention_entropy_mean",
    "attention_entropy_min",
    "attention_sink_first_token",
    "attention_sink_domination",
    "attention_sink_ratio",
  ]);
  if (spec.source_kind === "attention" || attentionReductions.has(reduction)) {
    add("b", "batch-example index");
    add("h", "attention-head index");
    add("q", "query-token position: the token that is attending");
    add("k", "key-token position: the token being attended to");
    add("T", "sequence length (number of token positions)");
    if (reduction.includes("entropy")) {
      add("H_{b,h,q}", "entropy of the attention distribution for batch item b, head h, and query q");
      add("X̃", "attention probabilities after flooring each value at 10⁻¹² for a stable logarithm");
    }
    if (reduction === "attention_entropy_mean") {
      add(
        "E_{b,h,q}",
        "arithmetic average over batch examples b, attention heads h, and query positions q",
        "\\mathbb{E}_{b,h,q}",
      );
    }
    if (reduction === "attention_sink_first_token") {
      add(
        "E_{b,h,q=1,…,T−1}",
        "arithmetic average over batch examples b, heads h, and non-initial query positions q = 1 through T − 1",
        "\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}",
      );
    }
    if (reduction === "attention_sink_domination") {
      add(
        "E_{b,h,q=0,…,T−1}",
        "arithmetic average over batch examples b, heads h, and all query positions q = 0 through T − 1",
        "\\mathbb{E}_{b,h,\\,q=0,\\ldots,T-1}",
      );
    }
    if (reduction === "attention_sink_ratio") {
      add(
        "E_{b,h,q=1,…,T−1}",
        "for T ≥ 2, the arithmetic average over batch examples b, heads h, and non-initial query positions q = 1 through T − 1",
        "\\mathbb{E}_{b,h,\\,q=1,\\ldots,T-1}",
      );
      add(
        "E_{b,h,q}",
        "for T = 1, the arithmetic average over batch examples b, heads h, and the only query position q",
        "\\mathbb{E}_{b,h,q}",
      );
    }
  }

  if (reduction === "massive_neuron_fraction") {
    add("b", "batch-example index");
    add("τ", "token-position index within the sequence (not the chart’s cumulative-learning-rate τ)");
    add("f", "feature or neuron index");
    add("j", "feature index used in the RMS sum");
    add("B", "batch size");
    add("T", "sequence length");
    add("F", "number of features or neurons");
    add("a_f", "largest absolute activation produced by feature f over all batch items and token positions");
  }
  if (["massive_activation_peak_ratio", "massive_activation_outlier_fraction",
       "massive_neuron_fraction"].includes(reduction)) {
    add("r", "root-mean-square (RMS) magnitude used as the typical activation scale");
  }

  if (tensorOps.some((op) => parseOpName(op) === "center") &&
      SAMPLE_AXIS_SOURCES.has(spec.source_kind)) {
    add("b", "batch-example index; E_b averages across the batch dimension");
  }
  if (tensorOps.some((op) => parseOpName(op) === "normalize") &&
      FEATURE_AXIS_SOURCES.has(spec.source_kind)) {
    add("f", "feature index; ‖·‖₂,f is the L2 norm along the feature dimension");
  }

  if (temporalOps.length) {
    add("t⁻", "previous observation for which the required input value is finite");
  }
  if (temporalOps.some((op) => ["slope", "rolling_std"].includes(parseOpName(op)))) {
    add("u_j", "j-th scalar value in the current temporal window, ordered from oldest to newest");
    add("n", "number of finite scalar observations currently available in that window");
    add("j", "position within the temporal window");
  }
  if (temporalOps.some((op) => parseOpName(op) === "slope")) {
    add("D", "sum of squared temporal-index deviations used as the slope denominator");
    add("j̄, ū", "means of the temporal indices and scalar values in the current window");
  }
  if (temporalOps.some((op) => parseOpName(op) === "curvature")) {
    add("u_0,u_1,u_2", "the latest three finite scalar observations, from oldest to newest");
  }

  return entries;
}

function buildSpecPlainDescription(spec) {
  if (!spec) return "";
  const place = modulePlaceEn(spec);
  const source = sourceObjectEn(spec);
  const { tensorOps, temporalOps } = pipelineOpsForSpec(spec);
  const transform = transformClauseEn(tensorOps, spec);
  const reduction = reductionMeaningEn(spec.reduction, spec.source_kind);
  const temporalNames = temporalOps.map(parseOpName);

  if (
    spec.source_kind === "weight" &&
    spec.reduction === "l2_norm" &&
    temporalNames.join("|") === "delta|ema"
  ) {
    return (
      `Tracks ${source} at ${place}. It smooths the change in overall magnitude ` +
      "between available observations."
    );
  }

  if (isDegenerateCenteredMean(spec, tensorOps)) {
    return (
      `Tracks ${source} at ${place}. Batch-centering makes its overall mean zero apart ` +
      "from numerical noise; the chart shows the recent slope of that residual."
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
  const lines = [`X_t=${sourceCallTex(spec)}`];
  let x = "X_t";
  const { tensorOps, temporalOps } = pipelineOpsForSpec(spec);
  tensorOps.forEach((op, index) => {
    const next = `X_t^{(${index + 1})}`;
    lines.push(`${next}=${applyTransformTex(op, x, spec)}`);
    x = next;
  });
  const renderedReduction = applyReductionTex(spec.reduction, x);
  const scalar =
    typeof renderedReduction === "string" ? renderedReduction : renderedReduction.main;
  const scalarDetails =
    typeof renderedReduction === "string" ? [] : renderedReduction.details || [];
  const tex = buildFormulaLines(lines, scalar, scalarDetails, temporalOps);
  const title = observableDisplayLabel(spec, temporalOps);
  return {
    tex,
    title,
    description: buildSpecPlainDescription(spec),
    notation: buildSpecNotation(spec),
  };
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
