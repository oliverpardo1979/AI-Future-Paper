const COLORS = {
  blue: "#1f5d88",
  teal: "#0d7a6a",
  orange: "#c95735",
  gold: "#a77212",
  violet: "#725aa8",
  ink: "#172225",
};

const REGIME_LABELS = {
  complements: "Complementos",
  cobb_douglas: "Cobb–Douglas",
  gross_substitutes: "Sustitutos brutos",
};

const DEFAULTS = {
  alpha: 0.33,
  omega_x: 0.2,
  sigma_xl: 1,
  n: 0.012,
  delta: 0.05,
  discount: 0.04,
  xi: 1,
  omega_m: 0.35,
  sigma_hm: 2,
  phi: 0.65,
  eta: 0.45,
  chi: 0.0983886742632607,
  capital_initial: 2.5093528013468305,
  capability_initial: 1,
  population_initial: 1,
  horizon: 2000,
  terminal_z: 0.2899267966577116,
  points: 181,
  tolerance: 0.00005,
  high_method: "fixed_horizon",
};

const els = {
  form: document.querySelector("#parameter-form"),
  benchmarkSelect: document.querySelector("#benchmark-select"),
  loadBenchmark: document.querySelector("#load-benchmark"),
  reset: document.querySelector("#reset-button"),
  cancel: document.querySelector("#cancel-button"),
  simulate: document.querySelector("#simulate-button"),
  download: document.querySelector("#download-button"),
  conditionPanel: document.querySelector("#condition-panel"),
  kicker: document.querySelector("#result-kicker"),
  title: document.querySelector("#results-title"),
  status: document.querySelector("#solver-status"),
  statusTitle: document.querySelector("#status-title"),
  statusDetail: document.querySelector("#status-detail"),
  metricRegime: document.querySelector("#metric-regime"),
  metricHorizon: document.querySelector("#metric-horizon"),
  metricInterest: document.querySelector("#metric-interest"),
  metricLaborShare: document.querySelector("#metric-labor-share"),
  verdict: document.querySelector("#diagnostic-verdict"),
  diagnostics: document.querySelector("#diagnostic-grid"),
  diagnosticNote: document.querySelector("#diagnostic-note"),
};

const charts = [
  {
    canvas: document.querySelector("#labor-allocation-chart"),
    legend: document.querySelector("#labor-allocation-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["production_labor_population_share", "Producción L/N", COLORS.teal],
      ["human_research_share", "Investigación H/N", COLORS.blue],
    ],
  },
  {
    canvas: document.querySelector("#labor-income-chart"),
    legend: document.querySelector("#labor-income-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["production_labor_share", "Trabajo en producción / Y", COLORS.teal],
      ["aggregate_labor_share", "Ingreso laboral total / Y", COLORS.blue],
    ],
  },
  {
    canvas: document.querySelector("#ai-services-chart"),
    legend: document.querySelector("#ai-services-legend"),
    format: formatAxis,
    series: [
      ["log_capability_change", "Capacidad A", COLORS.ink],
      ["log_inference_compute_per_capita_change", "Cómputo operativo U/N", COLORS.teal],
      ["log_ai_services_per_capita_change", "Servicios X/N", COLORS.blue],
    ],
  },
  {
    canvas: document.querySelector("#production-levels-chart"),
    legend: document.querySelector("#production-levels-legend"),
    format: formatAxis,
    series: [
      ["log_capital_per_capita_change", "Capital K/N", COLORS.violet],
      ["log_service_composite_per_capita_change", "Compuesto Z/N", COLORS.teal],
      ["log_output_per_capita_change", "Producto Y/N", COLORS.blue],
    ],
  },
  {
    canvas: document.querySelector("#household-levels-chart"),
    legend: document.querySelector("#household-levels-legend"),
    format: formatAxis,
    series: [
      ["log_consumption_per_capita_change", "Consumo C/N", COLORS.teal],
      ["log_wage_change", "Salario real w", COLORS.orange],
    ],
  },
  {
    canvas: document.querySelector("#growth-chart"),
    legend: document.querySelector("#growth-legend"),
    scale: 100,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["capability_growth", "Crecimiento A", COLORS.ink],
      ["output_per_capita_growth", "Crecimiento y/N", COLORS.blue],
      ["consumption_per_capita_growth", "Crecimiento c/N", COLORS.teal],
    ],
  },
  {
    canvas: document.querySelector("#factor-prices-chart"),
    legend: document.querySelector("#factor-prices-legend"),
    scale: 100,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["wage_growth", "Crecimiento del salario", COLORS.orange],
      ["net_interest", "Interés neto", COLORS.violet],
    ],
  },
  {
    canvas: document.querySelector("#automation-shares-chart"),
    legend: document.querySelector("#automation-shares-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["ai_share", "Servicios de IA en Z", COLORS.blue],
      ["automated_research_share", "Máquinas en E", COLORS.orange],
    ],
  },
  {
    canvas: document.querySelector("#allocation-chart"),
    legend: document.querySelector("#allocation-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["consumption_share", "Consumo C/Y", COLORS.gold],
      ["investment_share", "Inversión I/Y", COLORS.violet],
      ["inference_share", "Inferencia ξU/Y", COLORS.teal],
      ["research_resource_share", "Investigación automatizada ξM/Y", COLORS.orange],
    ],
  },
  {
    canvas: document.querySelector("#ai-price-chart"),
    legend: document.querySelector("#ai-price-legend"),
    format: formatAxis,
    series: [
      ["log_ai_price_change", "Precio del servicio pX", COLORS.blue],
      ["log_ai_marginal_cost_change", "Costo marginal ξ/A", COLORS.orange],
    ],
  },
  {
    canvas: document.querySelector("#markup-chart"),
    legend: document.querySelector("#markup-legend"),
    floor: 1,
    format: formatAxis,
    series: [
      ["ai_markup", "Markup pX/(ξ/A)", COLORS.ink],
    ],
  },
  {
    canvas: document.querySelector("#developer-value-chart"),
    legend: document.querySelector("#developer-value-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["ai_profit_share", "Beneficios operativos / Y", COLORS.gold],
    ],
  },
  {
    canvas: document.querySelector("#frontier-value-chart"),
    legend: document.querySelector("#frontier-value-legend"),
    format: formatAxis,
    transform: (value) => Math.log(value),
    series: [
      ["shadow_capability_to_output", "ln(qA/Y)", COLORS.violet],
      ["shadow_capability_to_capital", "ln(qA/K)", COLORS.blue],
    ],
  },
  {
    canvas: document.querySelector("#research-levels-chart"),
    legend: document.querySelector("#research-levels-legend"),
    format: formatAxis,
    series: [
      ["log_human_research_per_capita_change", "Investigadores H/N", COLORS.blue],
      ["log_automated_research_per_capita_change", "Máquinas M/N", COLORS.orange],
      ["log_effective_research_per_capita_change", "Investigación efectiva E/N", COLORS.violet],
    ],
  },
  {
    canvas: document.querySelector("#research-composition-chart"),
    legend: document.querySelector("#research-composition-legend"),
    scale: 100,
    floor: 0,
    format: (value) => `${formatAxis(value)}%`,
    series: [
      ["automated_research_share", "Participación de M en E", COLORS.orange],
    ],
  },
  {
    canvas: document.querySelector("#research-ratio-chart"),
    legend: document.querySelector("#research-ratio-legend"),
    format: formatAxis,
    transform: (value) => Math.log(value),
    series: [
      ["human_machine_ratio", "ln(H/M)", COLORS.ink],
    ],
  },
];

let benchmarkData = null;
let currentResult = null;
let worker = null;
let requestSequence = 0;

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function readParameters() {
  const values = {};
  document.querySelectorAll("[data-param]").forEach((input) => {
    if (input.tagName === "SELECT") {
      values[input.dataset.param] = input.value;
      return;
    }
    const raw = number(input.value);
    values[input.dataset.param] = input.hasAttribute("data-percent") ? raw / 100 : raw;
  });
  values.points = Math.round(values.points);
  return values;
}

function setParameters(values) {
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    if (!(key in values)) return;
    const displayed = input.hasAttribute("data-percent") ? values[key] * 100 : values[key];
    input.value = String(displayed);
  });
  renderConditions();
}

function renderConditions() {
  const p = readParameters();
  const omegaL = 1 - p.omega_x;
  const dCd = 1 - p.phi - p.eta * p.omega_m * p.omega_x / omegaL;
  const dAi = 1 - p.phi - p.eta * p.omega_x / omegaL;
  const highCap = 1 / p.alpha;
  const maintained = dCd > 0;
  const validInputs = Object.entries(p).every(([, value]) => typeof value === "string" || Number.isFinite(value));
  let className = "";
  let title = "Restricciones mantenidas satisfechas";
  let detail = "DCD > 0 y σHM >= 1. La simulación todavía debe pasar la auditoría numérica.";

  if (!validInputs || p.omega_x <= 0 || p.omega_x >= 1 || p.alpha <= 0 || p.alpha >= 1) {
    className = "condition-fail";
    title = "Revise los parámetros";
    detail = "Hay valores vacíos o fuera del dominio económico.";
  } else if (!maintained || p.sigma_hm < 1 || p.discount <= p.n) {
    className = "condition-fail";
    title = "Fuera del dominio mantenido";
    detail = "El paper estudia σHM = 1 y σHM > 1, y mantiene DCD > 0 y ρ > n. El solver rechazará esta corrida.";
  } else if (p.sigma_xl > 1 && p.sigma_xl >= highCap) {
    className = "condition-fail";
    title = "Elasticidad demasiado alta para esta rama";
    detail = `Para el algoritmo actual se requiere 1 < σXL < 1/α = ${highCap.toFixed(3)}.`;
  } else if (p.sigma_xl > 1) {
    className = "condition-warn";
    title = "Régimen de sustitutos brutos";
    detail = p.high_method === "free_boundary"
      ? "La frontera libre es experimental y puede tardar varios minutos. Pruebe primero el horizonte fijo."
      : "Se calcula una rama truncada al horizonte elegido; no se impone una senda balanceada terminal.";
  } else if (p.sigma_xl < 1) {
    className = "condition-warn";
    title = "Régimen de complementariedad";
    detail = "La solución de contorno suele ser lenta en el navegador, especialmente con horizontes largos.";
  }

  els.conditionPanel.className = `condition-panel ${className}`.trim();
  els.conditionPanel.innerHTML = `
    <div class="condition-title"><span>${title}</span><span>σXL ${relation(p.sigma_xl, 1)} 1</span></div>
    <div class="condition-values">
      <span>DCD<strong>${formatNumber(dCd, 4)}</strong></span>
      <span>DAI<strong>${formatNumber(dAi, 4)}</strong></span>
    </div>
    <p>${detail}</p>
  `;
}

function relation(left, right) {
  if (!Number.isFinite(left)) return "?";
  if (Math.abs(left - right) < 1e-8) return "=";
  return left < right ? "<" : ">";
}

function setStatus(kind, title, detail) {
  els.status.className = `solver-status status-${kind}`;
  els.statusTitle.textContent = title;
  els.statusDetail.textContent = detail;
}

function loadSelectedBenchmark() {
  if (!benchmarkData) return;
  const key = els.benchmarkSelect.value;
  const scenario = benchmarkData.scenarios[key];
  if (!scenario) return;
  const result = {
    source: "benchmark",
    label: scenario.label,
    diagnostics: scenario.diagnostics,
    series: scenario.series,
    parameters: {
      ...DEFAULTS,
      sigma_xl: scenario.sigma_xl,
      sigma_hm: scenario.sigma_hm ?? DEFAULTS.sigma_hm,
      horizon: scenario.diagnostics.duration || DEFAULTS.horizon,
      terminal_z: scenario.series.at(-1)?.output_capital_ratio || DEFAULTS.terminal_z,
    },
  };
  currentResult = result;
  setParameters(result.parameters);
  renderResult(result);
}

function validateClientParameters(p) {
  const messages = [];
  for (const [key, value] of Object.entries(p)) {
    if (typeof value !== "string" && !Number.isFinite(value)) messages.push(`${key} debe ser numérico.`);
  }
  if (!(p.alpha > 0 && p.alpha < 1)) messages.push("α debe estar entre cero y uno.");
  if (!(p.omega_x > 0 && p.omega_x < 1)) messages.push("ωX debe estar entre cero y uno.");
  if (!(p.omega_m > 0 && p.omega_m < 1)) messages.push("ωM debe estar entre cero y uno.");
  if (p.sigma_hm < 1) messages.push("Se requiere σHM >= 1.");
  if (p.discount <= p.n) messages.push("Se requiere ρ > n.");
  const dCd = 1 - p.phi - p.eta * p.omega_m * p.omega_x / (1 - p.omega_x);
  if (dCd <= 0) messages.push("Se requiere DCD > 0.");
  if (p.sigma_xl > 1 && p.sigma_xl >= 1 / p.alpha) messages.push("La rama de alta sustitución requiere σXL < 1/α.");
  if (p.horizon <= 0 || p.points < 61) messages.push("Revise horizonte y puntos reportados.");
  return messages;
}

function startSimulation(event) {
  event.preventDefault();
  const parameters = readParameters();
  const errors = validateClientParameters(parameters);
  if (errors.length) {
    setStatus("fail", "Parámetros no admisibles", errors.join(" "));
    return;
  }

  stopWorker();
  requestSequence += 1;
  const requestId = requestSequence;
  worker = new Worker(new URL("./solver-worker.js", import.meta.url), { type: "module" });
  els.simulate.disabled = true;
  els.cancel.hidden = false;
  setStatus(
    "working",
    "Preparando el solver",
    "La primera corrida descarga Python científico en el navegador. La última trayectoria válida permanece visible."
  );

  worker.addEventListener("message", (message) => {
    const payload = message.data || {};
    if (payload.requestId !== requestId && payload.type !== "status") return;
    if (payload.type === "status") {
      setStatus("working", payload.title || "Calculando", payload.detail || "Resolviendo el sistema de contorno…");
      return;
    }
    if (payload.type === "result") {
      finishWorker();
      if (payload.result.errors?.length) {
        setStatus("fail", "La corrida no produjo una trayectoria", payload.result.errors.join(" "));
        return;
      }
      const result = { ...payload.result, source: "custom", label: `Simulación propia · σXL = ${parameters.sigma_xl}` };
      if (!result.diagnostics?.passed) {
        setStatus("fail", "La trayectoria no pasó la auditoría", result.diagnostics?.interpretation || "Conserve el benchmark y ajuste horizonte, tolerancia o parámetros.");
        renderDiagnostics(result.diagnostics || {});
        return;
      }
      currentResult = result;
      renderResult(result);
      return;
    }
    if (payload.type === "error") {
      finishWorker();
      setStatus("fail", "Error del solver", payload.message || "No fue posible completar la corrida.");
    }
  });

  worker.addEventListener("error", (error) => {
    finishWorker();
    setStatus("fail", "No arrancó el solver", error.message || "Revise la conexión y vuelva a intentar.");
  });

  if (parameters.sigma_xl > 1 && benchmarkData) {
    const closest = Object.values(benchmarkData.scenarios)
      .filter((scenario) => scenario.sigma_xl > 1)
      .sort((left, right) => Math.abs(left.sigma_xl - parameters.sigma_xl) - Math.abs(right.sigma_xl - parameters.sigma_xl))[0];
    if (closest?.series) parameters.warm_start = closest.series;
  }
  worker.postMessage({ type: "simulate", requestId, parameters });
}

function stopWorker() {
  if (worker) worker.terminate();
  worker = null;
}

function finishWorker() {
  stopWorker();
  els.simulate.disabled = false;
  els.cancel.hidden = true;
}

function cancelSimulation() {
  if (!worker) return;
  finishWorker();
  setStatus("warn", "Cálculo cancelado", "No se cambió la última trayectoria válida.");
}

function renderResult(result) {
  const diagnostics = result.diagnostics || {};
  const series = result.series || [];
  if (!series.length) return;
  const final = series.at(-1);
  els.kicker.textContent = result.source === "benchmark" ? "Published benchmark" : "Custom canonical branch";
  els.title.textContent = result.label || "Trayectoria de equilibrio";
  els.metricRegime.textContent = REGIME_LABELS[diagnostics.regime] || diagnostics.regime || "—";
  els.metricHorizon.textContent = `${formatNumber(final.time, 0)} años`;
  els.metricInterest.textContent = formatPercent(final.net_interest);
  els.metricLaborShare.textContent = formatPercent(final.aggregate_labor_share);

  if (diagnostics.passed) {
    const prefix = result.source === "benchmark" ? "Benchmark validado" : "Rama canónica validada";
    setStatus("pass", prefix, "La trayectoria pasó los controles numéricos reportados. Esto no certifica optimalidad global ni unicidad.");
  } else {
    setStatus("fail", "Auditoría no superada", diagnostics.interpretation || "No interprete esta senda como equilibrio.");
  }

  renderDiagnostics(diagnostics);
  charts.forEach((chart) => drawChart(chart, series));
  els.download.disabled = false;
}

function renderDiagnostics(diagnostics) {
  const items = [
    ["Colocación", diagnostics.collocation_residual, "residual"],
    ["Dinámica", diagnostics.dynamic_residual, "residual"],
    ["Estática", diagnostics.static_residual, "residual"],
    ["Frontera", diagnostics.endpoint_residual, "residual"],
    ["Margen SOC", diagnostics.minimum_monopoly_margin, "number"],
    ["DCD", diagnostics.d_cd, "number"],
    ["DAI", diagnostics.d_ai, "number"],
    ["Interior", diagnostics.interior, "boolean"],
  ];
  if (diagnostics.estimated_singularity_time != null) {
    items.push(["Singularidad extrapolada", diagnostics.estimated_singularity_time, "years"]);
  }

  els.diagnostics.innerHTML = items.map(([label, value, kind]) => `
    <article><span>${label}</span><strong>${formatDiagnostic(value, kind)}</strong></article>
  `).join("");
  els.verdict.textContent = diagnostics.passed ? "Pasa" : "No pasa";
  els.verdict.className = `verdict ${diagnostics.passed ? "pass" : "fail"}`;
  els.diagnosticNote.textContent = diagnostics.interpretation ||
    "Los residuos se reportan sobre la malla mostrada y la solución de colocación.";
}

function formatDiagnostic(value, kind) {
  if (kind === "boolean") return value ? "Sí" : "No";
  if (!Number.isFinite(Number(value))) return "—";
  if (kind === "years") return `${formatNumber(value, 1)} años`;
  if (kind === "residual") return Math.abs(value) === 0 ? "0" : Number(value).toExponential(2);
  return formatNumber(value, 5);
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("es-CO", { maximumFractionDigits: digits }).format(value);
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 2 }).format(value);
}

function formatAxis(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return value.toExponential(1);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function finitePoints(rows, field, scale, transform = (value) => value) {
  return rows
    .map((row) => {
      const raw = Number(row[field]);
      return { x: Number(row.time), y: transform(raw) * scale };
    })
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

function niceBounds(minimum, maximum) {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return [-1, 1];
  if (minimum === maximum) {
    const pad = Math.max(Math.abs(minimum) * 0.08, 0.1);
    return [minimum - pad, maximum + pad];
  }
  const pad = (maximum - minimum) * 0.08;
  return [minimum - pad, maximum + pad];
}

function drawChart(config, rows) {
  const canvas = config.canvas;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const margin = { top: 12, right: 13, bottom: 38, left: 60 };
  const xTickCount = width < 320 ? 2 : 4;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const scale = config.scale || 1;
  const plotted = config.series.map(([field, label, color]) => ({
    field,
    label,
    color,
    points: finitePoints(rows, field, scale, config.transform),
  }));
  const all = plotted.flatMap((line) => line.points);
  if (!all.length) return;
  const xMin = Math.min(...all.map((point) => point.x));
  const xMax = Math.max(...all.map((point) => point.x));
  let yMin = Math.min(...all.map((point) => point.y));
  let yMax = Math.max(...all.map((point) => point.y));
  [yMin, yMax] = niceBounds(yMin, yMax);
  if (Number.isFinite(config.floor)) yMin = Math.max(yMin, config.floor);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const xPos = (value) => margin.left + (value - xMin) / xRange * plotWidth;
  const yPos = (value) => margin.top + (yMax - value) / yRange * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#637075";
  ctx.strokeStyle = "#ddd8cd";
  ctx.lineWidth = 1;

  for (let index = 0; index <= 4; index += 1) {
    const value = yMin + yRange * index / 4;
    const y = yPos(value);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(config.format(value), margin.left - 8, y);
  }

  for (let index = 0; index <= xTickCount; index += 1) {
    const value = xMin + xRange * index / xTickCount;
    const x = xPos(value);
    ctx.fillStyle = "#637075";
    ctx.textAlign = index === 0 ? "left" : index === xTickCount ? "right" : "center";
    ctx.textBaseline = "top";
    ctx.fillText(formatAxis(value), x, height - margin.bottom + 10);
  }
  ctx.textAlign = "right";
  ctx.fillText("años", width - margin.right, height - 10);

  plotted.forEach((line) => {
    if (!line.points.length) return;
    ctx.beginPath();
    line.points.forEach((point, index) => {
      const x = xPos(point.x);
      const y = yPos(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
  });

  config.legend.innerHTML = plotted.map((line) => `
    <span class="legend-item"><i class="legend-swatch" style="--color:${line.color}"></i>${line.label}</span>
  `).join("");
}

function downloadCsv() {
  if (!currentResult?.series?.length) return;
  const rows = currentResult.series;
  const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const escape = (value) => {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `equilibrium-${(currentResult.label || "simulation").replace(/[^a-z0-9]+/gi, "-").toLowerCase()}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function initialize() {
  renderConditions();
  els.download.disabled = true;
  try {
    const response = await fetch("./data/benchmarks.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    benchmarkData = await response.json();
    loadSelectedBenchmark();
  } catch (error) {
    setStatus("fail", "No cargaron los benchmarks", `${error.message}. La simulación propia sigue disponible.`);
  }
}

els.form.addEventListener("submit", startSimulation);
els.form.addEventListener("input", renderConditions);
els.form.addEventListener("change", renderConditions);
els.loadBenchmark.addEventListener("click", loadSelectedBenchmark);
els.benchmarkSelect.addEventListener("change", loadSelectedBenchmark);
els.reset.addEventListener("click", () => setParameters(DEFAULTS));
els.cancel.addEventListener("click", cancelSimulation);
els.download.addEventListener("click", downloadCsv);

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (currentResult?.series) charts.forEach((chart) => drawChart(chart, currentResult.series));
  }, 120);
});

initialize();
