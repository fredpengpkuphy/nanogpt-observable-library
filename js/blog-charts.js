(function () {
  "use strict";

  const COLORS = {
    ink: "#101920",
    grid: "rgba(237, 242, 245, 0.13)",
    tick: "rgba(237, 242, 245, 0.62)",
    label: "rgba(237, 242, 245, 0.82)",
    ice: "#9fd4e4",
    sand: "#e0c08a",
    coral: "#e89278",
    green: "#9ac9ad",
  };

  let chartData = null;
  let resizeTimer = null;

  function smooth(values, radius) {
    return values.map((_, index) => {
      const start = Math.max(0, index - radius);
      const end = Math.min(values.length - 1, index + radius);
      let total = 0;
      let count = 0;
      for (let i = start; i <= end; i += 1) {
        const value = Number(values[i]);
        if (!Number.isFinite(value)) continue;
        total += value;
        count += 1;
      }
      return count ? total / count : null;
    });
  }

  function formatStep(value) {
    if (value === 0) return "0";
    return `${Math.round(value / 1000)}k`;
  }

  function prepareCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, rect.width);
    const height = Math.max(280, rect.height);
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
  }

  function drawChart(canvas, steps, series, options) {
    const { context: ctx, width, height } = prepareCanvas(canvas);
    const compact = width < 560;
    const plot = {
      left: compact ? 43 : 58,
      right: width - (compact ? 12 : 22),
      top: compact ? 38 : 48,
      bottom: height - (compact ? 39 : 46),
    };
    const xMin = options.xMin || 0;
    const xMax = options.xMax || 100000;
    const yMin = options.yMin;
    const yMax = options.yMax;
    const xAt = (value) => plot.left + ((value - xMin) / (xMax - xMin)) * (plot.right - plot.left);
    const yAt = (value) => plot.bottom - ((value - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.ink;
    ctx.fillRect(0, 0, width, height);

    (options.bands || []).forEach((band) => {
      const left = xAt(band.start);
      const right = xAt(band.end);
      ctx.fillStyle = band.fill;
      ctx.fillRect(left, plot.top, right - left, plot.bottom - plot.top);
      ctx.fillStyle = band.text;
      ctx.font = `700 ${compact ? 10 : 11}px Nunito, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(band.label, (left + right) / 2, plot.top + 18);
    });

    const xTicks = compact ? [0, 50000, 100000] : [0, 25000, 50000, 75000, 100000];
    const yTicks = options.yTicks || [2, 3, 4, 5, 6];
    ctx.lineWidth = 1;
    ctx.font = `${compact ? 10 : 11}px Nunito, sans-serif`;

    xTicks.forEach((tick) => {
      const x = xAt(tick);
      ctx.strokeStyle = COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.fillStyle = COLORS.tick;
      ctx.textAlign = "center";
      ctx.fillText(formatStep(tick), x, plot.bottom + 20);
    });

    yTicks.forEach((tick) => {
      if (tick < yMin || tick > yMax) return;
      const y = yAt(tick);
      ctx.strokeStyle = COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.tick;
      ctx.textAlign = "right";
      ctx.fillText(String(tick), plot.left - 10, y + 4);
    });

    (options.transitions || []).forEach((step) => {
      const x = xAt(step);
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "rgba(237, 242, 245, 0.28)";
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.restore();
    });

    series.forEach((item) => {
      const values = smooth(item.values, item.smoothingRadius === undefined ? 2 : item.smoothingRadius);
      ctx.save();
      ctx.globalAlpha = item.opacity === undefined ? 1 : item.opacity;
      ctx.strokeStyle = item.color;
      ctx.lineWidth = item.width || 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      if (item.dash) ctx.setLineDash(item.dash);
      ctx.beginPath();
      let started = false;
      values.forEach((value, index) => {
        const step = Number(steps[index]);
        if (!Number.isFinite(step) || !Number.isFinite(value)) return;
        const x = xAt(step);
        const y = yAt(value);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      if (item.endDot) {
        const lastIndex = Math.min(steps.length, values.length) - 1;
        const lastValue = values[lastIndex];
        ctx.fillStyle = item.color;
        ctx.beginPath();
        ctx.arc(xAt(steps[lastIndex]), yAt(lastValue), 3.25, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });

    ctx.strokeStyle = "rgba(237, 242, 245, 0.34)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.bottom);
    ctx.lineTo(plot.right, plot.bottom);
    ctx.stroke();

    ctx.fillStyle = COLORS.label;
    ctx.font = `600 ${compact ? 10 : 11}px Nunito, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("training step", (plot.left + plot.right) / 2, height - 9);
    ctx.save();
    ctx.translate(13, (plot.top + plot.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("mean entropy", 0, 0);
    ctx.restore();
  }

  function renderFocusChart() {
    const canvas = document.getElementById("focusChart");
    if (!canvas) return;
    drawChart(
      canvas,
      chartData.steps,
      [{
        label: "Baseline Block 1",
        values: chartData.runs.baseline.layers["1"],
        color: COLORS.ice,
        width: 3.2,
        endDot: true,
      }],
      {
        yMin: 1.75,
        yMax: 6.1,
        yTicks: [2, 3, 4, 5, 6],
        transitions: [12000, 50000],
        bands: [
          { start: 1500, end: 12000, label: "≈ 4 plateau", fill: "rgba(224, 192, 138, 0.09)", text: COLORS.sand },
          { start: 18000, end: 50000, label: "≈ 2 plateau", fill: "rgba(159, 212, 228, 0.07)", text: COLORS.ice },
          { start: 50000, end: 100000, label: "toward 2.5", fill: "rgba(154, 201, 173, 0.06)", text: COLORS.green },
        ],
      },
    );
  }

  function renderLayerChart() {
    const canvas = document.getElementById("layerChart");
    if (!canvas) return;
    const mutedColors = [
      "#86949d", "#8c8798", "#778d8a", "#998b78", "#7f899d", "#768f9a",
      "#8a8390", "#6f8980", "#918778", "#778898", "#828c86", "#8d8181",
    ];
    const layers = chartData.runs.baseline.layers;
    const series = Object.keys(layers)
      .filter((layer) => layer !== "1")
      .map((layer) => ({
        label: `Block ${layer}`,
        values: layers[layer],
        color: mutedColors[Number(layer) % mutedColors.length],
        width: 1.35,
        opacity: 0.62,
      }));
    series.push({
      label: "Block 1",
      values: layers["1"],
      color: COLORS.ice,
      width: 3.4,
      opacity: 1,
      endDot: true,
    });
    drawChart(canvas, chartData.steps, series, {
      yMin: 1.75,
      yMax: 6.1,
      yTicks: [2, 3, 4, 5, 6],
    });
  }

  function renderSetupChart() {
    const canvas = document.getElementById("setupChart");
    if (!canvas) return;
    drawChart(
      canvas,
      chartData.steps,
      [
        {
          label: "12-layer baseline",
          values: chartData.runs.baseline.layers["1"],
          color: COLORS.ice,
          width: 3,
          endDot: true,
        },
        {
          label: "6-layer nanoGPT",
          values: chartData.runs["6_layers_nanogpt"].layers["1"],
          color: COLORS.sand,
          width: 2.6,
          endDot: true,
        },
        {
          label: "No learning-rate warmup",
          values: chartData.runs.no_learning_rate_warmup.layers["1"],
          color: COLORS.coral,
          width: 2.6,
          endDot: true,
        },
      ],
      {
        yMin: 1.75,
        yMax: 6.1,
        yTicks: [2, 3, 4, 5, 6],
      },
    );
  }

  function renderAll() {
    if (!chartData) return;
    renderFocusChart();
    renderLayerChart();
    renderSetupChart();
  }

  function showError() {
    document.querySelectorAll(".chart-status").forEach((status) => {
      status.hidden = false;
      status.textContent = "The processed curve data could not be loaded.";
    });
  }

  const fontReady = document.fonts && document.fonts.ready
    ? document.fonts.ready
    : Promise.resolve();

  Promise.all([
    fetch("data/blog-attention-entropy.json").then((response) => {
      if (!response.ok) throw new Error(`Curve data request failed: ${response.status}`);
      return response.json();
    }),
    fontReady,
  ])
    .then(([data]) => {
      chartData = data;
      document.querySelectorAll(".chart-status").forEach((status) => {
        status.hidden = true;
      });
      renderAll();
    })
    .catch((error) => {
      console.error(error);
      showError();
    });

  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(renderAll, 120);
  });
})();
