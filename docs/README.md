# AI Growth Lab

Static GitHub Pages companion for the equilibrium model in *Automating
Innovation*. It has two distinct modes:

1. It loads the five published and audited benchmark paths from
   `data/benchmarks.json` immediately.
2. It runs the browser solver in a Web Worker through Pyodide, NumPy, and SciPy
   when a reader supplies new parameters or initial conditions.

The interface deliberately distinguishes a numerical branch that satisfies the
reported canonical equilibrium system from a proof of global dynamic optimality,
existence, or uniqueness. Failed custom paths never replace the last valid result.

## Rebuild the published benchmarks

From the repository root:

```text
python scripts/build_web_benchmarks.py
```

The builder reads the canonical CSV outputs in `numerical/` and writes the
browser-ready JSON file. Commit that JSON so the public page does not need a
server or a Python backend.

## Local preview

Serve the repository rather than opening `index.html` directly, because browsers
restrict module workers and `fetch` on `file://` URLs:

```text
python -m http.server 8000 --directory docs
```

Then visit `http://localhost:8000/`.

## Deployment

`.github/workflows/pages.yml` publishes `docs/` with GitHub Pages after relevant
pushes to `main`. The repository Pages source must be set to **GitHub Actions**.

