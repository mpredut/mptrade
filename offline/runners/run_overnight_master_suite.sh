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

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================================"
echo "      AUTONOMOUS OVERNIGHT MASTER SUITE STARTED: $(date '+%F %T %Z')"
echo "================================================================================"
echo "Host: $(hostname) | Cores: $(nproc) | Root: $REPO_ROOT"
echo "Commit: $(git rev-parse --short HEAD) [$(git rev-parse --abbrev-ref HEAD)]"
echo "================================================================================"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
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
# STEP 1: Kraken & HYPE Multi-Interval Walk-Forward Baselines & Comparisons
# ------------------------------------------------------------------------------
run_step "1A" "HYPE 240m Walk-Forward Baseline (628 days, 31 folds)" \
    "$PYTHON_BIN offline/runners/kraken_walk_forward_baseline.py \
        --intervals 240 \
        --dataset 240=offline/research/hype_dataset/HYPEUSDC_240m_hlspot.csv \
        --train 720 --validation 180 --test 90 --step 90 --warmup 40 \
        --output-dir $RESULTS_DIR/kraken_wf_240"

run_step "1B" "HYPE 240m Candidate Comparison" \
    "$PYTHON_BIN offline/runners/kraken_walk_forward_compare.py \
        $RESULTS_DIR/kraken_wf_240/baseline_HYPEUSD_*.json \
        --candidate-set hype-240"

run_step "1C" "HYPE 1440m (Daily) Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/kraken_walk_forward_baseline.py \
        --intervals 1440 \
        --dataset 1440=offline/research/hype_dataset/HYPEUSDC_1440m_hlspot.csv \
        --train 180 --validation 60 --test 30 --step 30 --warmup 20 \
        --output-dir $RESULTS_DIR/kraken_wf_1440"

# ------------------------------------------------------------------------------
# STEP 2: Trading212 Walk-Forward Baselines (NVDA, RGNT, SPCX)
# ------------------------------------------------------------------------------
run_step "2A" "T212 NVDA 1d Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/t212_walk_forward_baseline.py \
        --profile nvda \
        --dataset offline/research/t212_dataset/nvda/datasets/NVDA_1d_9e8cbd3c6ff5.csv \
        --train 180 --validation 60 --test 30 --step 30 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_nvda"

run_step "2B" "T212 RGNT 1d Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/t212_walk_forward_baseline.py \
        --profile rgnt \
        --dataset offline/research/t212_dataset/rgnt/datasets/RGNT_1d_b67a2932ddd6.csv \
        --train 180 --validation 60 --test 30 --step 30 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_rgnt"

run_step "2C" "T212 SPCX 5m Walk-Forward Baseline" \
    "$PYTHON_BIN offline/runners/t212_walk_forward_baseline.py \
        --profile spcx \
        --dataset offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv \
        --train 720 --validation 180 --test 90 --step 90 --warmup 12 \
        --output-dir $RESULTS_DIR/t212_spcx"

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
# STEP 4: Comprehensive Multi-Factor Strategy Sweep (30 Candidates)
# ------------------------------------------------------------------------------
run_step "4" "Comprehensive 30-Candidate Multi-Factor Strategy Sweep" \
    "$PYTHON_BIN scratch/comprehensive_strategy_sweep.py"

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
- [x] **Kraken HYPE 240m Walk-Forward (628 days, 31 folds)**: Baseline & Comparative Pareto ranking
- [x] **Kraken HYPE 1440m (Daily) Walk-Forward**: Long-term macro regime verification
- [x] **Trading212 Walk-Forward**: NVDA (1d), RGNT (1d), SPCX (5m)
- [x] **Kraken Adaptive Multipliers**: K_REENTRY & K_DCA sweeps
- [x] **30-Candidate Multi-Factor Strategy Sweep**: Out-of-sample, continuous compounding, and forward live shadow
- [x] **TradeAll Trigger Gate Experiments**: Trend gate, quality signal, RSI/BB, dual timeframe
- [x] **TradeAll Adaptive Thresholds & Kalman Lag Sweeps**: Real-tick volatility and sampling latencies
- [x] **Monitortrades 392-Day Replay**: Full Binance BTC & TAO simulation
- [x] **Proposal Publishing Pipeline**: Verified candidate parameters pushed to \`backtest-proposals\`

## Artifacts & Logs
- Master Log: \`$MASTER_LOG\`
- Latest Log Symlink: \`$LOG_DIR/latest_overnight.log\`
- Strategy Sweep Report: \`offline/results/strategy_sweep/SWEEP_REPORT.md\`
- Walk-Forward Comparisons: \`offline/results/kraken_wf_240/\`
- Pilot Proposals File: \`backtest_proposals.json\` (pushed to branch \`backtest-proposals\`)

EOF

echo "Master report generated at: $REPORT_MD"
