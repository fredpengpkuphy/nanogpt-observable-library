(function () {
  const GROUPS = [
    {
      key: "data",
      label: "Data & run",
      fields: [
        ["dataset", "Dataset"],
        ["init_from", "Initialization"],
      ],
    },
    {
      key: "model",
      label: "Model",
      fields: [
        ["n_layer", "Layers"],
        ["n_head", "Attention heads"],
        ["n_embd", "Embedding width"],
        ["block_size", "Context length"],
        ["vocab_size", "Vocabulary size"],
        ["dropout", "Dropout"],
        ["bias", "Bias"],
      ],
    },
    {
      key: "optimizer",
      label: "Optimizer",
      fields: [
        ["name", "Type"],
        ["learning_rate", "Max learning rate"],
        ["betas", "Betas"],
        ["weight_decay", "Weight decay"],
        ["grad_clip", "Gradient clip"],
      ],
    },
    {
      key: "lr_schedule",
      label: "LR schedule",
      fields: [
        ["decay", "Decay"],
        ["warmup_iters", "Warmup steps"],
        ["lr_decay_iters", "Decay horizon"],
        ["min_lr", "Minimum learning rate"],
      ],
    },
    {
      key: "batching",
      label: "Batching",
      fields: [
        ["micro_batch_size_per_gpu", "Micro-batch / GPU"],
        ["gradient_accumulation_steps", "Global grad accumulation"],
        ["tokens_per_iteration", "Tokens / iteration"],
      ],
    },
  ];

  function clone(value) {
    if (Array.isArray(value)) return value.map(clone);
    if (!value || typeof value !== "object") return value;
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
  }

  function merge(base, overrides) {
    const out = clone(base || {});
    for (const [key, value] of Object.entries(overrides || {})) {
      if (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        out[key] &&
        typeof out[key] === "object" &&
        !Array.isArray(out[key])
      ) {
        out[key] = merge(out[key], value);
      } else {
        out[key] = clone(value);
      }
    }
    return out;
  }

  function displayValue(value) {
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return value.toLocaleString("en-US");
    return String(value);
  }

  function appendText(parent, tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = text;
    parent.appendChild(el);
    return el;
  }

  function render(container, baseConfig, run) {
    if (!container || !baseConfig) return false;
    const config = merge(baseConfig, run?.config_overrides);
    container.innerHTML = "";

    const intro = document.createElement("div");
    intro.className = "training-config-intro";
    appendText(intro, "strong", "", config.name || "Training setup");
    if (run?.setup_note) appendText(intro, "p", "", run.setup_note);
    if (Number.isInteger(config.recorded_through_step)) {
      appendText(
        intro,
        "p",
        "training-config-recording",
        `Available recording: steps 0–${displayValue(config.recorded_through_step)}`,
      );
    }
    if (config.source_url) {
      const link = document.createElement("a");
      link.href = config.source_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "View the upstream nanoGPT config";
      intro.appendChild(link);
    }
    container.appendChild(intro);

    const groups = document.createElement("div");
    groups.className = "training-config-groups";
    for (const groupDef of GROUPS) {
      const values = config[groupDef.key];
      if (!values) continue;
      const section = document.createElement("section");
      section.className = "training-config-group";
      appendText(section, "h4", "", groupDef.label);
      const list = document.createElement("dl");
      for (const [key, label] of groupDef.fields) {
        if (values[key] === undefined || values[key] === null) continue;
        const row = document.createElement("div");
        appendText(row, "dt", "", label);
        appendText(row, "dd", "", displayValue(values[key]));
        list.appendChild(row);
      }
      section.appendChild(list);
      groups.appendChild(section);
    }
    container.appendChild(groups);
    return true;
  }

  window.TrainingConfig = { merge, render };
})();
