#!/usr/bin/env bash
# run_overnight_master_suite.sh — Autonomous Overnight Master Backtest & Research Runner on DEV.
set -uo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
cd "$ROOT"

LOG_DIR="$ROOT/logs/overnight_runs"
RESULTS_DIR="$ROOT/offline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

LOCK_FILE="/tmp/mptrade_overnight_master_suite.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "[overnight_master_suite] Another instance is already running (lock: $LOCK_FILE) — exiting." >&2; exit 0; }

MASTER_LOG="$LOG_DIR/master_suite_$(date '+%Y%m%d_%H%M%S').log"
REPORT_MD="$RESULTS_DIR/OVERNIGHT_MASTER_REPORT.md"
ls -1t "$LOG_DIR"/master_suite_20*.log 2>/dev/null | tail -n +21 | xargs -r rm -f || true
ln -sf "$MASTER_LOG" "$LOG_DIR/latest_overnight.log"
ln -sf "$MASTER_LOG" "$LOG_DIR/master_suite_current.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================================"
echo "  AUTONOMOUS OVERNIGHT MASTER SUITE STARTED: $(date '+%F %T %Z')"
echo "  Host: $(hostname) | Cores: $(nproc) | Root: $ROOT | Commit: $(git rev-parse --short HEAD)"
echo "================================================================================"

START_SEC=$(date +%s); PASSED_STEPS=0; FAILED_STEPS=0; STEP_REPORT_LINES=""

run_step() {  # $1=id $2=name $3=cmd
    local id="$1" name="$2" cmd="$3" t0 rc=0 dt
    echo -e "\n--------------------------------------------------------------------------------\n[STEP $id] START: $name ($(date '+%T')) | CMD: $cmd\n--------------------------------------------------------------------------------"
    t0=$(date +%s); eval "$cmd" || rc=$?; dt=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        PASSED_STEPS=$((PASSED_STEPS + 1))
        STEP_REPORT_LINES+="| \`$id\` | $name | PASS | ${dt}s |"$'\n'
        echo "[STEP $id] COMPLETED: $name in ${dt}s"
    else
        FAILED_STEPS=$((FAILED_STEPS + 1))
        STEP_REPORT_LINES+="| \`$id\` | $name | **FAIL (exit $rc)** | ${dt}s |"$'\n'
        echo "[WARN] Step $id finished with exit code $rc (${dt}s)"
    fi
}

run_wf_pair() {  # $1=base_id $2=comp_id $3=label $4=out_subdir $5=glob $6=wf_args [$7=cmp_args]
    local out="$RESULTS_DIR/$4"
    run_step "$1" "$3 Walk-Forward Baseline" "$PYTHON_BIN offline/runners/walk_forward.py $6 --output-dir $out"
    run_step "$2" "$3 Candidate Comparison" "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \$(ls -t $out/$5 2>/dev/null | head -1) ${7:-}"
}

# STEP 0: Market Intelligence 4-Pillar Backtest, Chop Benchmark & Numerical Parity
run_step "0A" "4-Pillar Market Intelligence Backtest (SSOT symbols)" "$PYTHON_BIN offline/research/intelligence_backtest.py"
run_step "0B" "Chop Regime & Dynamic Weight Benchmark (407 days)" "$PYTHON_BIN offline/research/chop_weight_backtest.py"
run_step "0C" "Numerical Parity & Zero-Regression Suite (30,000+ validations)" "$PYTHON_BIN offline/research/verify_optimizations_parity.py"

# STEP 1 & 2: Multi-Venue Walk-Forward Baselines & Candidate Comparisons
run_wf_pair "1A" "1B"      "HYPE 240m (628d, 31 folds)" "kraken_wf_240"  "baseline_HYPEUSD_*.json"               "--venue kraken --pair HYPEUSD --intervals 240 --dataset 240=offline/research/hype_dataset/HYPEUSDC_240m_hlspot.csv --train 720 --validation 180 --test 90 --step 90 --warmup 40" "--candidate-set hype-240"
run_step    "1C"           "HYPE Financial Central & Stress Promotion Gate Comparison" "$PYTHON_BIN offline/runners/kraken_financial_compare.py"
run_wf_pair "2A" "2A_COMP" "T212 NVDA 1d"               "t212_nvda"      "{,nvda/}baseline_NVDA_*.json"          "--venue t212 --profile nvda --dataset offline/research/t212_dataset/nvda/datasets/NVDA_1d_9e8cbd3c6ff5.csv --train 180 --validation 60 --test 30 --step 30 --warmup 12"
run_wf_pair "2B" "2B_COMP" "T212 RGNT 1d"               "t212_rgnt"      "{,rgnt/}baseline_RGNT_*.json"          "--venue t212 --profile rgnt --dataset offline/research/t212_dataset/rgnt/datasets/RGNT_1d_b67a2932ddd6.csv --train 90 --validation 30 --test 20 --step 20 --warmup 12"
run_wf_pair "2C" "2C_COMP" "T212 SPCX 5m"               "t212_spcx"      "{,spcx/}baseline_SPCX_*.json"          "--venue t212 --profile spcx --interval 5m --dataset offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv --train 720 --validation 180 --test 90 --step 90 --warmup 12"
run_wf_pair "2D" "2D_COMP" "Hyperliquid HYPE 240m"      "hyperliquid_wf" "HYPE_240m_baseline.json"               "--venue hyperliquid --coin HYPE --direction long --intervals 240"
run_wf_pair "2E" "2E_COMP" "Binance BTCUSDC 240m"       "binance_wf"     "BTCUSDC_240m_baseline.json"            "--venue binance --symbol BTCUSDC --intervals 240"

# STEP 4..10: Strategy Sweeps, Monitortrades Replay, Proposals, Continuous Grids & Pytest Regression
run_step "4A" "Comprehensive 30-Candidate Multi-Factor Strategy Sweep" "$PYTHON_BIN offline/research/comprehensive_strategy_sweep.py"
run_step "4B" "Deep Parabolic Surge & Bear-Bounce Multi-Variable Grid Sweep (969 Candidates)" "$PYTHON_BIN offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py --fee-pct 0.26"
run_step "4C" "Hybrid Re-Entry Ablation Suite (11 Crypto Assets & US Equities)" "$PYTHON_BIN offline/research/hybrid_reentry/run_suite.py && $PYTHON_BIN offline/research/hybrid_reentry/run_equities.py && $PYTHON_BIN offline/research/hybrid_reentry/report.py"
run_step "4D" "MHDSL Reentry Replay Comparisons" "$PYTHON_BIN offline/research/compare_mhdsl_reentry_replay.py"
run_step "7A" "Monitortrades Replay Backtest on Binance Historical Ticks" "$PYTHON_BIN offline/research/monitortrades_backtest/run_replay_backtest.py"
run_step "7B" "Monitortrades MHDSL Structural Break Variant Comparison" "$PYTHON_BIN offline/research/monitortrades_backtest/compare_sl_variants.py"
run_step "8"  "Scheduled Pilot Proposals Generation & Worktree Push" "bash offline/runners/run_backtest_cycle.sh local"
run_step "9A" "Kraken Continuous Multi-Asset Grid (11 assets since 2018)" "bash offline/research/kraken_continuous_grid/run_all.sh"
run_step "9B" "Trading212 Continuous Multi-Asset Factorial Grid (16 equities/ETFs, 54 variants)" "$PYTHON_BIN offline/research/t212_continuous_grid/fetch.py --missing-only && $PYTHON_BIN offline/research/t212_continuous_grid/grid.py && $PYTHON_BIN offline/research/t212_continuous_grid/report.py"
run_step "10" "Full DEV Test Regression Suite (pytest tests/)" "bash offline/runners/run_dev_regression.sh run"

TOTAL_ELAPSED=$(( $(date +%s) - START_SEC ))
DURATION_FMT="$((TOTAL_ELAPSED / 3600))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $((TOTAL_ELAPSED % 60))s"
echo -e "\n================================================================================\n  AUTONOMOUS OVERNIGHT MASTER SUITE FINISHED: $(date '+%F %T %Z') ($DURATION_FMT | Passed: $PASSED_STEPS | Failed: $FAILED_STEPS)\n================================================================================"

cat << EOF > "$REPORT_MD"
# Autonomous Overnight Master Backtest & Research Report

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ') | Duration: $DURATION_FMT | Host: $(hostname) | Commit: $(git rev-parse --short HEAD)
Passed Steps: $PASSED_STEPS | Failed Steps: $FAILED_STEPS

## Stage Execution Matrix
| Step | Module | Status | Duration |
| :--- | :--- | :--- | ---: |
${STEP_REPORT_LINES}
## Artifacts & Logs
- Master Log: \`$MASTER_LOG\` (\`$LOG_DIR/latest_overnight.log\`)
- Strategy & Surge Sweeps: \`offline/results/strategy_sweep/SWEEP_REPORT.md\`, \`offline/results/surge_sweep/DEEP_SURGE_SWEEP_REPORT.md\`
- Hybrid Re-entry & Grids: \`offline/results/hybrid_reentry/ablation_summary.json\`, \`offline/results/t212_continuous_grid/REPORT.md\`
- Walk-Forward & Proposals: \`offline/results/kraken_wf_240/\`, \`backtest_proposals.json\` (\`backtest-proposals\`)
EOF
echo "Master report generated at: $REPORT_MD"
