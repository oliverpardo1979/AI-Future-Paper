import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs";

let pyodidePromise;
let solverLoaded = false;
let activeRequestId = null;

function status(title, detail = "") {
  self.postMessage({ type: "status", requestId: activeRequestId, title, detail });
}

async function ensureSolver() {
  if (!pyodidePromise) {
    status("Preparing the scientific runtime", "The first run downloads Python, NumPy, and SciPy.");
    pyodidePromise = loadPyodide();
  }
  const pyodide = await pyodidePromise;
  if (!solverLoaded) {
    status("Loading numerical packages", "This happens once per browser session.");
    await pyodide.loadPackage(["numpy", "scipy"]);
    const solverUrl = new URL("./py/equilibrium_solver.py", import.meta.url);
    const response = await fetch(solverUrl);
    if (!response.ok) {
      throw new Error(`Could not load the equilibrium solver (${response.status}).`);
    }
    const source = await response.text();
    await pyodide.runPythonAsync(source);
    solverLoaded = true;
    status("Solver ready", "The equilibrium calculation can now begin.");
  }
  return pyodide;
}

self.onmessage = async (event) => {
  const { type, requestId, parameters } = event.data || {};
  if (type !== "simulate") return;
  activeRequestId = requestId;
  try {
    const pyodide = await ensureSolver();
    status(
      "Solving the equilibrium branch",
      parameters.sigma_xl > 1
        ? "Gross-substitution paths may take several minutes."
        : "The interface remains responsive while the boundary-value problem runs.",
    );
    pyodide.globals.set("_browser_parameters_json", JSON.stringify(parameters));
    const resultJson = await pyodide.runPythonAsync(
      "simulate_json(_browser_parameters_json)",
    );
    self.postMessage({
      type: "result",
      requestId,
      result: JSON.parse(resultJson),
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
