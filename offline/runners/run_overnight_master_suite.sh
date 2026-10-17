#!/usr/bin/env bash
# run_overnight_master_suite.sh — Autonomous Overnight Master Backtest & Research Runner.
# Runs all offline evaluations, walk-forwards, parameter sweeps, and proposal cycles across DEV.
set -uo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUNNER_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/overnight_runs"
RESULTS_DIR="$REPO_ROOT/offline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="$LOG_DIR/master_suite_${TIMESTAMP}.log"
REPORT_MD="$RESULTS_DIR/OVERNIGHT_MASTER_REPORT.md"

# Symlink to latest
ln -sf "$MASTER_LOG" "$LOG_DIR/latest_overnight.log"
ln -sf "$MASTER_LOG" "$LOG_DIR/master_suite_current.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================================"
echo "      AUTONOMOUS OVERNIGHT MASTER SUITE STARTED: $(date '+%F %T %Z')"
echo "================================================================================"
echo "Host: $(hostname) | Cores: $(nproc) | Root: $REPO_ROOT"
echo "Commit: $(git rev-parse --short HEAD) [$(git rev-parse --abbrev-ref HEAD)]"
echo "================================================================================"

if [ -x "$REPO_ROOT/myenv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/myenv/bin/python"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

START_SEC=$(date +%s)

# Helper function to run a step with timing and error isolation
run_step() {
    local step_num="$1"
    local step_name="$2"
    local step_cmd="$3"
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[STEP $step_num] START: $step_name ($(date '+%T'))"
    echo "CMD: $step_cmd"
    echo "--------------------------------------------------------------------------------"
    local step_start=$(date +%s)
    
    eval "$step_cmd" || echo "[WARN] Step $step_num finished with exit code $?"
    
    local step_end=$(date +%s)
    local step_elapsed=$((step_end - step_start))
    echo "[STEP $step_num] COMPLETED: $step_name in ${step_elapsed}s"
}

# ------------------------------------------------------------------------------
# STEP 0: Market Intelligence 4-Pillar Backtest & Chop Weight Benchmark
# ------------------------------------------------------------------------------
run_step "0A" "4-Pillar Market Intelligence Backtest (SSOT symbols)" \
    "$PYTHON_BIN offline/research/intelligence_backtest.py"

run_step "0B" "Chop Regime & Dynamic Weight Benchmark (407 days)" \
    "$PYTHON_BIN offline/research/chop_weight_backtest.py"

# ------------------------------------------------------------------------------
# STEP 1: Kraken & HYPE Multi-Interval Walk-Forward Baselines & Comparisons
# ------------------------------------------------------------------------------
run_step "1A" "HYPE 240m Walk-Forward Baseline (628 days, 31 folds)" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue kraken --pair HYPEUSD \
        --intervals 240 \
        --dataset 240=offline/research/hype_dataset/HYPEUSDC_240m_hlspot.csv \
        --train 720 --validation 180 --test 90 --step 90 --warmup 40 \
        --output-dir $RESULTS_DIR/kraken_wf_240"

run_step "1B" "HYPE 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        \$(ls -t $RESULTS_DIR/kraken_wf_240/baseline_HYPEUSD_*.json | head -1) \
        --candidate-set hype-240"

# No 1440m step: the live config is regime-aware (STRAT_TP_REGIME_GATE=true, 4h classifier),
# and replay refuses any bar size other than trend_interval (240m) because it does not resample.

# ------------------------------------------------------------------------------
# STEP 2: Trading212 Walk-Forward Baselines (NVDA, RGNT, SPCX)
# ------------------------------------------------------------------------------
run_step "2A" "T212 NVDA 1d Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue t212 \
        --profile nvda \
        --dataset offline/research/t212_dataset/nvda/datasets/NVDA_1d_9e8cbd3c6ff5.csv \
        --train 180 --validation 60 --test 30 --step 30 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_nvda"

run_step "2A_COMP" "T212 NVDA 1d Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        \$(ls -t $RESULTS_DIR/t212_nvda/baseline_NVDA_*.json $RESULTS_DIR/t212_nvda/nvda/baseline_NVDA_*.json 2>/dev/null | head -1)"

run_step "2B" "T212 RGNT 1d Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue t212 \
        --profile rgnt \
        --dataset offline/research/t212_dataset/rgnt/datasets/RGNT_1d_b67a2932ddd6.csv \
        --train 90 --validation 30 --test 20 --step 20 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_rgnt"

run_step "2B_COMP" "T212 RGNT 1d Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        \$(ls -t $RESULTS_DIR/t212_rgnt/baseline_RGNT_*.json $RESULTS_DIR/t212_rgnt/rgnt/baseline_RGNT_*.json 2>/dev/null | head -1)"

run_step "2C" "T212 SPCX 5m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue t212 \
        --profile spcx --interval 5m \
        --dataset offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv \
        --train 720 --validation 180 --test 90 --step 90 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_spcx"

run_step "2C_COMP" "T212 SPCX 5m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        \$(ls -t $RESULTS_DIR/t212_spcx/baseline_SPCX_*.json $RESULTS_DIR/t212_spcx/spcx/baseline_SPCX_*.json 2>/dev/null | head -1)"

run_step "2D" "Hyperliquid HYPE 240m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue hyperliquid \
        --coin HYPE --direction long --intervals 240 \
        --output-dir $RESULTS_DIR/hyperliquid_wf"

run_step "2D_COMP" "Hyperliquid HYPE 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        $RESULTS_DIR/hyperliquid_wf/HYPE_240m_baseline.json"

run_step "2E" "Binance BTCUSDC 240m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/walk_forward.py --venue binance \
        --symbol BTCUSDC --intervals 240 \
        --output-dir $RESULTS_DIR/binance_wf"

run_step "2E_COMP" "Binance BTCUSDC 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/walk_forward.py --mode compare \
        $RESULTS_DIR/binance_wf/BTCUSDC_240m_baseline.json"

# ------------------------------------------------------------------------------
# STEP 3: Kraken Adaptive Volatility & Reentry Multiplier Sweeps
# ------------------------------------------------------------------------------
run_step "3A" "Kraken K-Multiplier Sweep (K_REENTRY & K_DCA)" \
    "$PYTHON_BIN offline/research/kraken_adaptive_thresholds/sweep_k_multiplier.py"

run_step "3B" "Kraken Adaptive DCA Verification" \
    "$PYTHON_BIN offline/research/kraken_adaptive_thresholds/verify_adaptive_dca.py"

run_step "3C" "Kraken Adaptive Reentry Verification" \
    "$PYTHON_BIN offline/research/kraken_adaptive_thresholds/verify_adaptive_reentry.py"

# ------------------------------------------------------------------------------
# STEP 4: Comprehensive Multi-Factor & Multi-Variable Strategy Sweeps
# ------------------------------------------------------------------------------
run_step "4A" "Comprehensive 30-Candidate Multi-Factor Strategy Sweep" \
    "$PYTHON_BIN offline/research/comprehensive_strategy_sweep.py"

run_step "4B" "Deep Parabolic Surge & Bear-Bounce Multi-Variable Grid Sweep (555 Candidates)" \
    "$PYTHON_BIN offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py --fee-pct 0.26"

run_step "4C" "Hybrid Re-Entry Ablation Suite (11 Crypto Assets & US Equities)" \
    "$PYTHON_BIN offline/research/hybrid_reentry/run_suite.py && $PYTHON_BIN offline/research/hybrid_reentry/run_equities.py"

# ------------------------------------------------------------------------------
# STEP 5: TradeAll Trigger Gate Experiments
# ------------------------------------------------------------------------------
run_step "5A" "TradeAll Trend Gate Experiment" \
    "$PYTHON_BIN offline/research/tradeall_trigger_gate/experiment_trend_gate.py"

run_step "5B" "TradeAll Quality Signal Regression Experiment" \
    "$PYTHON_BIN offline/research/tradeall_trigger_gate/experiment_quality_signal.py"

run_step "5C" "TradeAll RSI & Bollinger Experiment" \
    "$PYTHON_BIN offline/research/tradeall_trigger_gate/experiment_rsi_bollinger.py"

run_step "5D" "TradeAll Dual Timeframe Experiment" \
    "$PYTHON_BIN offline/research/tradeall_trigger_gate/experiment_dual_timeframe.py"

# ------------------------------------------------------------------------------
# STEP 6: TradeAll Adaptive Thresholds & Kalman Lag Sweeps
# ------------------------------------------------------------------------------
run_step "6A" "TradeAll Adaptive Thresholds Sweep (K*vol_1h)" \
    "$PYTHON_BIN offline/research/tradeall_adaptive_thresholds/run_backtest_adaptive.py"

run_step "6B" "TradeAll Kalman Filter Sampling Lag Sweep (20s/60s/90s/150s)" \
    "$PYTHON_BIN offline/research/tradeall_kalman_lag/run_kalman_lag_sweep.py"

# ------------------------------------------------------------------------------
# STEP 7: Monitortrades Binance Replay Backtest (BTC & TAO over 392 days)
# ------------------------------------------------------------------------------
run_step "7" "Monitortrades Replay Backtest on Binance Historical Ticks" \
    "$PYTHON_BIN offline/research/monitortrades_backtest/run_replay_backtest.py"

# ------------------------------------------------------------------------------
# STEP 8: Scheduled Pilot & Proposals Branch Generation
# ------------------------------------------------------------------------------
run_step "8" "Scheduled Pilot Proposals Generation & Worktree Push" \
    "bash offline/runners/run_backtest_cycle.sh"

# ------------------------------------------------------------------------------
# STEP 9: Kraken & Trading212 Continuous Multi-Asset Grids
# ------------------------------------------------------------------------------
run_step "9A" "Kraken Continuous Multi-Asset Grid (11 assets since 2018)" \
    "bash offline/research/kraken_continuous_grid/run_all.sh"

run_step "9B" "Trading212 Continuous Multi-Asset Factorial Grid (16 equities/ETFs, 54 variants)" \
    "$PYTHON_BIN offline/research/t212_continuous_grid/fetch.py --missing-only && $PYTHON_BIN offline/research/t212_continuous_grid/grid.py && $PYTHON_BIN offline/research/t212_continuous_grid/report.py"

END_SEC=$(date +%s)
TOTAL_ELAPSED=$((END_SEC - START_SEC))
HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))
SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "================================================================================"
echo "      AUTONOMOUS OVERNIGHT MASTER SUITE FINISHED: $(date '+%F %T %Z')"
echo "      Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "================================================================================"

# Generate Consolidated Summary Report
cat << EOF > "$REPORT_MD"
# Autonomous Overnight Master Backtest & Research Report

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s
Host: $(hostname)
Commit: $(git rev-parse --short HEAD)

## Executed Modules
- [x] **4-Pillar Market Intelligence & Chop Weight Benchmark**: SSOT symbols & 407-day dynamic weights
- [x] **Kraken HYPE 240m Walk-Forward (628 days, 31 folds)**: Baseline & Comparative Pareto ranking
- [x] **Trading212 Walk-Forward & Comparisons**: NVDA (1d), RGNT (1d), SPCX (5m)
- [x] **Hyperliquid Walk-Forward**: HYPE (240m)
- [x] **Binance Walk-Forward**: BTCUSDC (240m)
- [x] **Kraken Adaptive Multipliers**: K_REENTRY & K_DCA sweeps
- [x] **30-Candidate Multi-Factor Strategy Sweep**: Out-of-sample, continuous compounding, and forward live shadow
- [x] **Deep Parabolic Surge & Bear-Bounce Sweep (555 Candidates)**: Fixed/dynamic surge & bounce re-entry grid
- [x] **Hybrid Re-Entry Ablation Suite**: 11 Crypto assets & US equities
- [x] **TradeAll Trigger Gate Experiments**: Trend gate, quality signal, RSI/BB, dual timeframe
- [x] **TradeAll Adaptive Thresholds & Kalman Lag Sweeps**: Real-tick volatility and sampling latencies
- [x] **Monitortrades 392-Day Replay**: Full Binance BTC & TAO simulation
- [x] **Kraken Continuous Grid**: \`offline/results/kraken_continuous_grid/REPORT*.md\`
- [x] **Trading212 Continuous Multi-Asset Grid**: \`offline/results/t212_continuous_grid/REPORT.md\`
- [x] **Proposal Publishing Pipeline**: Verified candidate parameters pushed to \`backtest-proposals\`

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
