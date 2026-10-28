#!/usr/bin/env bash
# run_overnight_master_suite.sh — Autonomous Overnight Master Backtest & Research Runner.
# Runs all offline evaluations, walk-forwards, parameter sweeps, and proposal cycles across DEV.
set -uo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="$ROOT"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/overnight_runs"
RESULTS_DIR="$REPO_ROOT/offline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

LOCK_FILE="/tmp/mptrade_overnight_master_suite.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "[overnight_master_suite] Another instance is already running (lock: $LOCK_FILE) — exiting." >&2; exit 0; }

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="$LOG_DIR/master_suite_${TIMESTAMP}.log"
REPORT_MD="$RESULTS_DIR/OVERNIGHT_MASTER_REPORT.md"

ls -1t "$LOG_DIR"/master_suite_20*.log 2>/dev/null | tail -n +21 | xargs -r rm -f || true
ln -sf "$MASTER_LOG" "$LOG_DIR/latest_overnight.log"
ln -sf "$MASTER_LOG" "$LOG_DIR/master_suite_current.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================================"
echo "      AUTONOMOUS OVERNIGHT MASTER SUITE STARTED: $(date '+%F %T %Z')"
echo "      Host: $(hostname) | Cores: $(nproc) | Root: $REPO_ROOT | Commit: $(git rev-parse --short HEAD) [$(git rev-parse --abbrev-ref HEAD)]"
echo "================================================================================"

START_SEC=$(date +%s)
PASSED_STEPS=0
FAILED_STEPS=0
STEP_REPORT_LINES=""

run_step() {
    local step_num="$1" step_name="$2" step_cmd="$3" step_start rc=0 step_elapsed
    echo -e "\n--------------------------------------------------------------------------------"
    echo "[STEP $step_num] START: $step_name ($(date '+%T')) | CMD: $step_cmd"
    echo "--------------------------------------------------------------------------------"
    step_start=$(date +%s)
    eval "$step_cmd" || rc=$?
    step_elapsed=$(( $(date +%s) - step_start ))
    if [ "$rc" -eq 0 ]; then
        PASSED_STEPS=$((PASSED_STEPS + 1))
        STEP_REPORT_LINES="${STEP_REPORT_LINES}| \`$step_num\` | $step_name | PASS | ${step_elapsed}s |"$'\n'
        echo "[STEP $step_num] COMPLETED: $step_name in ${step_elapsed}s"
    else
        FAILED_STEPS=$((FAILED_STEPS + 1))
        STEP_REPORT_LINES="${STEP_REPORT_LINES}| \`$step_num\` | $step_name | **FAIL (exit $rc)** | ${step_elapsed}s |"$'\n'
        echo "[WARN] Step $step_num finished with exit code $rc (${step_elapsed}s)"
    fi
}

run_t212_wf() {  # $1=id $2=profile $3=ticker $4=label $5=dataset $6=extra_args
    local id="$1" prof="$2" sym="$3" label="$4" ds="$5" extra="$6" out="$RESULTS_DIR/t212_${prof}"
    run_step "$id" "T212 $label Walk-Forward Baseline" \
        "$PYTHON_BIN offline/runners/walk_forward.py --venue t212 --profile $prof --dataset $ds $extra --output-dir $out"
    run_step "${id}_COMP" "T212 $label Candidate Comparison" \
        "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \$(ls -t $out/baseline_${sym}_*.json $out/$prof/baseline_${sym}_*.json 2>/dev/null | head -1)"
}

# STEP 0: Market Intelligence 4-Pillar Backtest, Chop Benchmark & Numerical Parity
run_step "0A" "4-Pillar Market Intelligence Backtest (SSOT symbols)" "$PYTHON_BIN offline/research/intelligence_backtest.py"
run_step "0B" "Chop Regime & Dynamic Weight Benchmark (407 days)" "$PYTHON_BIN offline/research/chop_weight_backtest.py"
run_step "0C" "Numerical Parity & Zero-Regression Suite (30,000+ validations)" "$PYTHON_BIN offline/research/verify_optimizations_parity.py"

# STEP 1: Kraken & HYPE Multi-Interval Walk-Forward Baselines & Comparisons
run_step "1A" "HYPE 240m Walk-Forward Baseline (628 days, 31 folds)" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue kraken --pair HYPEUSD --intervals 240 --dataset 240=offline/research/hype_dataset/HYPEUSDC_240m_hlspot.csv --train 720 --validation 180 --test 90 --step 90 --warmup 40 --output-dir $RESULTS_DIR/kraken_wf_240"
run_step "1B" "HYPE 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \$(ls -t $RESULTS_DIR/kraken_wf_240/baseline_HYPEUSD_*.json | head -1) --candidate-set hype-240"
run_step "1C" "HYPE Financial Central & Stress Promotion Gate Comparison" "$PYTHON_BIN offline/runners/kraken_financial_compare.py"

# STEP 2: Trading212, Hyperliquid & Binance Walk-Forward Baselines
run_t212_wf "2A" "nvda" "NVDA" "NVDA 1d" "offline/research/t212_dataset/nvda/datasets/NVDA_1d_9e8cbd3c6ff5.csv" "--train 180 --validation 60 --test 30 --step 30 --warmup 12"
run_t212_wf "2B" "rgnt" "RGNT" "RGNT 1d" "offline/research/t212_dataset/rgnt/datasets/RGNT_1d_b67a2932ddd6.csv" "--train 90 --validation 30 --test 20 --step 20 --warmup 12"
run_t212_wf "2C" "spcx" "SPCX" "SPCX 5m" "offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv" "--interval 5m --train 720 --validation 180 --test 90 --step 90 --warmup 12"

run_step "2D" "Hyperliquid HYPE 240m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue hyperliquid --coin HYPE --direction long --intervals 240 --output-dir $RESULTS_DIR/hyperliquid_wf"
run_step "2D_COMP" "Hyperliquid HYPE 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare $RESULTS_DIR/hyperliquid_wf/HYPE_240m_baseline.json"
run_step "2E" "Binance BTCUSDC 240m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue binance --symbol BTCUSDC --intervals 240 --output-dir $RESULTS_DIR/binance_wf"
run_step "2E_COMP" "Binance BTCUSDC 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare $RESULTS_DIR/binance_wf/BTCUSDC_240m_baseline.json"

# STEP 4: Comprehensive Multi-Factor & Multi-Variable Strategy Sweeps
run_step "4A" "Comprehensive 30-Candidate Multi-Factor Strategy Sweep" "$PYTHON_BIN offline/research/comprehensive_strategy_sweep.py"
run_step "4B" "Deep Parabolic Surge & Bear-Bounce Multi-Variable Grid Sweep (969 Candidates)" "$PYTHON_BIN offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py --fee-pct 0.26"
run_step "4C" "Hybrid Re-Entry Ablation Suite (11 Crypto Assets & US Equities)" \
    "$PYTHON_BIN offline/research/hybrid_reentry/run_suite.py && $PYTHON_BIN offline/research/hybrid_reentry/run_equities.py && $PYTHON_BIN offline/research/hybrid_reentry/report.py"
run_step "4D" "MHDSL Reentry Replay Comparisons" "$PYTHON_BIN offline/research/compare_mhdsl_reentry_replay.py"

# STEP 7: Monitortrades Binance Replay Backtest (BTC & TAO over 392 days)
run_step "7A" "Monitortrades Replay Backtest on Binance Historical Ticks" "$PYTHON_BIN offline/research/monitortrades_backtest/run_replay_backtest.py"
run_step "7B" "Monitortrades MHDSL Structural Break Variant Comparison" "$PYTHON_BIN offline/research/monitortrades_backtest/compare_sl_variants.py"

# STEP 8: Scheduled Pilot & Proposals Branch Generation
run_step "8" "Scheduled Pilot Proposals Generation & Worktree Push" "bash offline/runners/run_backtest_cycle.sh"

# STEP 9: Kraken & Trading212 Continuous Multi-Asset Grids
run_step "9A" "Kraken Continuous Multi-Asset Grid (11 assets since 2018)" "bash offline/research/kraken_continuous_grid/run_all.sh"
run_step "9B" "Trading212 Continuous Multi-Asset Factorial Grid (16 equities/ETFs, 54 variants)" \
    "$PYTHON_BIN offline/research/t212_continuous_grid/fetch.py --missing-only && $PYTHON_BIN offline/research/t212_continuous_grid/grid.py && $PYTHON_BIN offline/research/t212_continuous_grid/report.py"

# STEP 10: Full DEV CI Pytest Regression Suite
run_step "10" "Full DEV Test Regression Suite (pytest tests/)" \
    "env -u BACKTEST_MODE -u REPLAY_MODE -u DISABLE_EXTERNAL_NOTIFICATIONS -u DISABLE_GEMINI_LLM bash offline/runners/run_dev_regression.sh"

TOTAL_ELAPSED=$(( $(date +%s) - START_SEC ))
HOURS=$((TOTAL_ELAPSED / 3600)); MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 )); SECONDS=$((TOTAL_ELAPSED % 60))

echo -e "\n================================================================================"
echo "      AUTONOMOUS OVERNIGHT MASTER SUITE FINISHED: $(date '+%F %T %Z')"
echo "      Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s | Passed: $PASSED_STEPS | Failed: $FAILED_STEPS"
echo "================================================================================"

cat << EOF > "$REPORT_MD"
# Autonomous Overnight Master Backtest & Research Report

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s
Host: $(hostname)
Commit: $(git rev-parse --short HEAD)
Passed Steps: $PASSED_STEPS | Failed Steps: $FAILED_STEPS

## Stage Execution Matrix
| Step | Module | Status | Duration |
| :--- | :--- | :--- | ---: |
${STEP_REPORT_LINES}
## Artifacts & Logs
- Master Log: \`$MASTER_LOG\`
- Latest Log Symlink: \`$LOG_DIR/latest_overnight.log\`
- Strategy Sweep Report: \`offline/results/strategy_sweep/SWEEP_REPORT.md\`
- Deep Surge Sweep Report: \`offline/results/surge_sweep/DEEP_SURGE_SWEEP_REPORT.md\`
- Hybrid Re-entry Summary: \`offline/results/hybrid_reentry/ablation_summary.json\`
- Walk-Forward Comparisons: \`offline/results/kraken_wf_240/\`
- Trading212 Grid Report: \`offline/results/t212_continuous_grid/REPORT.md\`
- Pilot Proposals File: \`backtest_proposals.json\` (pushed to branch \`backtest-proposals\`)
EOF
echo "Master report generated at: $REPORT_MD"
