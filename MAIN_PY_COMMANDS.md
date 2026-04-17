# main.py Command Reference

Quick command cheat sheet for running the Inventory Optimization agent from CLI.

## 0) Setup (Windows PowerShell)

```powershell
# from repo root
& ".\venv\Scripts\Activate.ps1"
```

## 1) Basic Runs

```powershell
# default data + default config + table output
python main.py

# explicit CSV input
python main.py --data data/inventory_mock.csv

# explicit JSON input
python main.py --data data/inventory_mock.json
```

## 2) Output Formats

```powershell
# table output (default)
python main.py --format table

# full JSON output to stdout
python main.py --format json
```

## 3) Output File and Report Control

```powershell
# write run output JSON to custom file
python main.py --output results/run_custom.json

# disable markdown report generation
python main.py --no-report
```

## 4) Agent Modes

```powershell
# deterministic mode (default)
python main.py --agent-mode deterministic

# hybrid mode
python main.py --agent-mode hybrid

# full mode
python main.py --agent-mode full
```

## 5) Scenario Overrides

```powershell
# one override
python main.py --scenario lead_time=10

# multiple overrides
python main.py --scenario lead_time=10 --scenario safety_stock=25 --scenario healthy_dos_min=12
```

## 6) Single-SKU Analysis

Use the included one-row file:

```powershell
python main.py --data data/inventory_single_sku.csv --format json
```

## 7) Single-SKU Analysis with gemma3:4b

Use the provided config that targets `gemma3:4b`:

```powershell
python main.py --data data/inventory_single_sku.csv --config config/thresholds_gemma3.yaml --format json --agent-mode deterministic
```

## 8) Useful Combined Examples

```powershell
# hybrid + JSON + custom output
python main.py --data data/inventory_mock.csv --agent-mode hybrid --format json --output results/hybrid_run.json

# full mode + scenario stress test
python main.py --data data/inventory_mock.csv --agent-mode full --scenario lead_time=14 --scenario safety_stock=40 --format json
```

## 9) Help

```powershell
python main.py -h
```

---

## Current Input/Config Files Added for Quick Testing

- `data/inventory_single_sku.csv` (single row: `SKU-001`)
- `config/thresholds_gemma3.yaml` (model set to `gemma3:4b`)
