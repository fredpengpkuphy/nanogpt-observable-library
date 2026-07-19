/**
 * Math definitions mirrored from reference.html (observable_lib).
 * Values are KaTeX bodies without surrounding $$.
 */
const SOURCE_DEFS = {
  weight: String.raw`x = \theta_t`,
  grad: String.raw`x = g_t = \nabla_\theta \mathcal{L}`,
  update: String.raw`x = u_t = \theta_t - \theta_{t-1}`,
  opt_m: String.raw`x = \hat{m}_t`,
  opt_v: String.raw`x = \hat{v}_t`,
  activation: String.raw`x = a_t`,
  preactivation: String.raw`x = \tilde{a}_t`,
  logits: String.raw`x = z_t`,
  attention: String.raw`x = A_t \in \mathbb{R}^{B\times H\times T\times T}`,
  gelu_activation: String.raw`x = \mathrm{GELU}(\cdot)`,
};

const TRANSFORM_DEFS = {
  identity: String.raw`\mathcal{T}(x)=x`,
  center: String.raw`\mathcal{T}(x)=x-\bar{x}_{\mathrm{sample}}`,
  abs: String.raw`\mathcal{T}(x)=|x|`,
  normalize: String.raw`\mathcal{T}(x)=\dfrac{x}{\|x\|_2+\varepsilon}`,
  square: String.raw`\mathcal{T}(x)=x^{2}`,
};

const REDUCTION_DEFS = {
  mean: String.raw`\mathcal{R}(x)=\mathbb{E}[x]`,
  std: String.raw`\mathcal{R}(x)=\mathrm{Std}(x)`,
  min: String.raw`\mathcal{R}(x)=\min x`,
  max: String.raw`\mathcal{R}(x)=\max x`,
  l1_norm: String.raw`\mathcal{R}(x)=\|x\|_1=\sum_i |x_i|`,
  l2_norm: String.raw`\mathcal{R}(x)=\|x\|_2=\sqrt{\sum_i x_i^{2}}`,
  rms: String.raw`\mathcal{R}(x)=\mathrm{RMS}(x)=\sqrt{\mathbb{E}[x^{2}]}`,
  max_abs: String.raw`\mathcal{R}(x)=\max_i |x_i|`,
  sparsity: String.raw`\mathcal{R}(x)=\dfrac{1}{N}\sum_i \mathbf{1}[|x_i|<10^{-6}]`,
  positive_fraction: String.raw`\mathcal{R}(x)=\dfrac{1}{N}\sum_i \mathbf{1}[x_i>0]`,
  entropy: String.raw`p_i=\dfrac{|x_i|}{\sum_j |x_j|},\;\mathcal{R}(x)=-\sum_i p_i\log p_i`,
  spectral_norm: String.raw`\mathcal{R}(W)=\sigma_{\max}(W)`,
  top_singular_value: String.raw`\mathcal{R}(W)=\sigma_{\max}(W)`,
  trace: String.raw`\mathcal{R}(W)=\mathrm{tr}(W)`,
  row_std_mean: String.raw`\mathcal{R}(W)=\dfrac{1}{r}\sum_{i=1}^{r}\mathrm{Std}(W_{i,:})`,
  col_std_mean: String.raw`\mathcal{R}(W)=\dfrac{1}{c}\sum_{j=1}^{c}\mathrm{Std}(W_{:,j})`,
  effective_rank: String.raw`\tilde{X}=X-\bar{X},\;p_k=\dfrac{\lambda_k}{\sum_j\lambda_j},\;\mathcal{R}=\exp\!\big(-\sum_k p_k\log p_k\big)`,
  attention_entropy_mean: String.raw`H_{b,h,q}=-\sum_k A_{b,h,q,k}\log A_{b,h,q,k},\;\mathcal{R}=\mathbb{E}_{b,h,q}[H]`,
  attention_entropy_min: String.raw`\mathcal{R}=\min_{b,h,q} H_{b,h,q}`,
  attention_sink_first_token: String.raw`\mathcal{R}=\mathbb{E}_{b,h,q\ge 1}[A_{b,h,q,0}]`,
  attention_sink_domination: String.raw`\bar{A}_k=\mathbb{E}_{b,h,q}[A_{b,h,q,k}],\;\mathcal{R}=\max_k\bar{A}_k`,
  attention_sink_ratio: String.raw`\mathcal{R}=T\cdot\mathbb{E}_{b,h,q\ge 1}[A_{b,h,q,0}]`,
  activation_rate: String.raw`\mathcal{R}(x)=\dfrac{1}{N}\sum_i\mathbf{1}[x_i>0]`,
  massive_activation_peak_ratio: String.raw`\mathcal{R}(x)=\dfrac{\max|x|}{\mathrm{RMS}(x)}`,
  massive_activation_outlier_fraction: String.raw`\mathcal{R}(x)=\dfrac{1}{N}\sum_i\mathbf{1}[|x_i|>3\,\mathrm{RMS}(x)]`,
  massive_neuron_fraction: String.raw`p_f=\max_{b,t}|x_{b,t,f}|,\;\mathcal{R}=\dfrac{1}{F}\sum_f\mathbf{1}[p_f>3\,\mathrm{RMS}(x)]`,
};

const TEMPORAL_DEFS = {
  identity: String.raw`y_t=s_t`,
  delta: String.raw`y_t=s_t-s_{t-1}`,
  ema: String.raw`y_t=\alpha\,y_{t-1}+(1-\alpha)\,s_t`,
  slope: String.raw`y_t=\mathrm{OLS\,slope}(\{s_{t-w+1},\ldots,s_t\})`,
  rolling_std: String.raw`y_t=\mathrm{Std}(s_{t-w+1},\ldots,s_t)`,
  curvature: String.raw`y_t=s_t-2s_{t-1}+s_{t-2}`,
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

/** Temporal op TeX, preserving alpha/window params when present. */
function temporalDefTex(raw) {
  const name = parseOpName(raw);
  const args = parseOpArgs(raw);
  if (name === "ema") {
    const alpha = parseNamedArg(args, "alpha");
    if (alpha != null) {
      const a = String(alpha);
      return String.raw`y_t=${a}\,y_{t-1}+(1-${a})\,s_t`;
    }
  }
  if (name === "slope" || name === "rolling_std") {
    const w = parseNamedArg(args, "window") || parseNamedArg(args, "w");
    if (w != null) {
      return name === "slope"
        ? String.raw`y_t=\mathrm{OLS\,slope}(\{s_{t-${w}+1},\ldots,s_t\})`
        : String.raw`y_t=\mathrm{Std}(s_{t-${w}+1},\ldots,s_t)`;
    }
  }
  return TEMPORAL_DEFS[name] || null;
}

function splitPipelineOps(transforms) {
  const tensorOps = [];
  const temporalOps = [];
  for (const raw of transforms || []) {
    const name = parseOpName(raw);
    if (TEMPORAL_DEFS[name] && !TRANSFORM_DEFS[name]) temporalOps.push(raw);
    else if (TRANSFORM_DEFS[name] || name === "identity") tensorOps.push(raw);
    else if (TEMPORAL_DEFS[name]) temporalOps.push(raw);
    else tensorOps.push(raw);
  }
  return { tensorOps, temporalOps };
}

function referenceAnchorForSpec(spec) {
  const r = spec?.reduction || "";
  if (r.startsWith("attention_")) return "reference.html#reductions-attention";
  if (r.startsWith("massive_") || r === "activation_rate") return "reference.html#reductions-activation";
  if (["spectral_norm", "top_singular_value", "trace", "row_std_mean", "col_std_mean", "effective_rank"].includes(r)) {
    return "reference.html#reductions-matrix";
  }
  return "reference.html#reductions-basic";
}

/** Build KaTeX-ready pieces for a selected spec. */
function buildSpecDefinitionLatex(spec) {
  if (!spec) return null;
  const pieces = [];
  const sourceTex = SOURCE_DEFS[spec.source_kind];
  if (sourceTex) pieces.push({ kind: "source", tex: sourceTex, name: spec.source_kind });

  const { tensorOps, temporalOps } = splitPipelineOps(spec.transforms || []);
  for (const raw of tensorOps) {
    const name = parseOpName(raw);
    const tex = TRANSFORM_DEFS[name];
    if (tex) pieces.push({ kind: "transform", tex, name: raw });
  }

  const redTex = REDUCTION_DEFS[spec.reduction];
  if (redTex) pieces.push({ kind: "reduction", tex: redTex, name: spec.reduction });
  else {
    const safe = String(spec.reduction).replace(/[^a-zA-Z0-9_]/g, "");
    pieces.push({ kind: "reduction", tex: `\\mathcal{R}_{\\mathrm{${safe}}}(x)`, name: spec.reduction });
  }

  for (const raw of temporalOps) {
    const tex = temporalDefTex(raw);
    if (tex) pieces.push({ kind: "temporal", tex, name: raw });
  }

  return pieces;
}
