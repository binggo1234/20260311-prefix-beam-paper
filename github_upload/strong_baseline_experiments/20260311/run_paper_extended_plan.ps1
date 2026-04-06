param(
    [ValidateSet("all", "main", "ablation", "strong_ablation", "strong_baselines", "strong_external", "strong_time", "strong_repeat", "strong_params", "robustness", "params", "layout", "figures")]
    [string]$Stage = "all",
    [string]$OutRoot = "outputs",
    [int]$RobustnessSeconds = 1200,
    [string]$IndustrialLargeCase = "data2",
    [string]$IndustrialMediumCase = "data6",
    [string]$PublicCase = "gcut6",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ProjectRoot
$PythonExe = "python"
$OutRootAbs = Join-Path $ProjectRoot $OutRoot

function Invoke-Task {
    param(
        [string]$Name,
        [string[]]$ArgList
    )
    $cmd = "python " + ($ArgList -join " ")
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    Write-Host $cmd
    if (-not $DryRun) {
        & $PythonExe @ArgList
        if ($LASTEXITCODE -ne 0) {
            throw "Task failed: $Name"
        }
    }
}

Set-Location $RepoRoot

if ($Stage -in @("all", "main")) {
    Invoke-Task -Name "industrial main 4-method" -ArgList @(
        "20260311/run_paper_experiments.py", "robustness",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_main_ind_all_1seed_300s",
        "--out-root", $OutRoot,
        "--variants", "baseline_greedy", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "public main 4-method" -ArgList @(
        "20260311/run_paper_experiments.py", "robustness",
        "--suite", "public_compatible",
        "--exp-name", "exp_main_pub_all_1seed_300s",
        "--out-root", $OutRoot,
        "--variants", "baseline_greedy", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "ablation")) {
    Invoke-Task -Name "industrial ablation" -ArgList @(
        "20260311/run_paper_experiments.py", "ablation",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_ablation_ind_3seed_600s",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "600"
    )

    Invoke-Task -Name "public ablation" -ArgList @(
        "20260311/run_paper_experiments.py", "ablation",
        "--suite", "public_compatible",
        "--exp-name", "exp_ablation_pub_1seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "strong_ablation")) {
    Invoke-Task -Name "industrial strong ablation" -ArgList @(
        "20260311/run_paper_experiments.py", "robustness",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_ablation_strong_ind",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_ref", "rh_mcts_full", "ours_nomacro", "ours_nolightrepair", "rh_mcts_prefix_beam",
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "600"
    )

    Invoke-Task -Name "public strong ablation" -ArgList @(
        "20260311/run_paper_experiments.py", "robustness",
        "--suite", "public_compatible",
        "--exp-name", "exp_ablation_strong_pub",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_ref", "rh_mcts_full", "ours_nomacro", "ours_nolightrepair", "rh_mcts_prefix_beam",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "strong_baselines", "strong_external")) {
    Invoke-Task -Name "industrial pilot strong baseline" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "compare",
        "--baseline-kind", "pilot",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_pilot_ind_1seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "public pilot strong baseline" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "compare",
        "--baseline-kind", "pilot",
        "--suite", "public_compatible",
        "--exp-name", "exp_strong_pilot_pub_1seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "industrial full beam strong baseline" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "compare",
        "--baseline-kind", "full_beam",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_full_beam_ind_1seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "public full beam strong baseline" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "compare",
        "--baseline-kind", "full_beam",
        "--suite", "public_compatible",
        "--exp-name", "exp_strong_full_beam_pub_1seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "strong_baselines", "strong_time")) {
    Invoke-Task -Name "industrial strong baseline budget scan" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "time_scan",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_budget_ind_3seed",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--budgets", "10", "30", "60", "120", "300"
    )

    Invoke-Task -Name "public strong baseline budget scan" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "time_scan",
        "--suite", "public_compatible",
        "--exp-name", "exp_strong_budget_pub10_1seed",
        "--out-root", $OutRoot,
        "--n-seeds", "1",
        "--seed0", "1000",
        "--budgets", "10", "30", "60", "120", "300"
    )
}

if ($Stage -in @("all", "strong_baselines", "strong_repeat")) {
    Invoke-Task -Name "industrial strong baseline repeatability" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "repeatability",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_repeat_ind_5seed_300s",
        "--out-root", $OutRoot,
        "--n-seeds", "5",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "strong_baselines", "strong_params")) {
    Invoke-Task -Name "industrial pilot parameter grid" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "param_grid",
        "--target", "pilot",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_param_pilot_ind",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "public pilot parameter grid" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "param_grid",
        "--target", "pilot",
        "--suite", "public_compatible",
        "--exp-name", "exp_strong_param_pilot_pub5",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "industrial full beam parameter grid" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "param_grid",
        "--target", "full_beam",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_strong_param_full_beam_ind",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "public full beam parameter grid" -ArgList @(
        "20260311/tools/run_strong_baseline_batches.py", "param_grid",
        "--target", "full_beam",
        "--suite", "public_compatible",
        "--exp-name", "exp_strong_param_full_beam_pub5",
        "--out-root", $OutRoot,
        "--n-seeds", "3",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "robustness")) {
    Invoke-Task -Name "industrial robustness 5seed 1200s" -ArgList @(
        "20260311/run_paper_experiments.py", "robustness",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_robust_ind_5seed_${RobustnessSeconds}s",
        "--out-root", $OutRoot,
        "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--n-seeds", "5",
        "--seed0", "1000",
        "--solver-time-limit-s", "$RobustnessSeconds"
    )
}

if ($Stage -in @("all", "params")) {
    Invoke-Task -Name "industrial nsim sensitivity" -ArgList @(
        "20260311/run_paper_experiments.py", "sensitivity",
        "--suite", "industrial_xlsx6",
        "--exp-name", "exp_nsim_ind",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_prefix_beam",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300",
        "--mcts-n-sim-values", "1", "2", "3", "5", "8"
    )

    Invoke-Task -Name "prefix depth grid" -ArgList @(
        "20260311/tools/run_prefix_param_grid.py",
        "--suite", "all_known",
        "--exp-name", "exp_prefix_depth_grid",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_prefix_beam",
        "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
        "--parameter", "PREFIX_BEAM_DEPTH",
        "--values", "1", "2", "3", "4", "5",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "prefix width grid" -ArgList @(
        "20260311/tools/run_prefix_param_grid.py",
        "--suite", "all_known",
        "--exp-name", "exp_prefix_width_grid",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_prefix_beam",
        "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
        "--parameter", "PREFIX_BEAM_WIDTH",
        "--values", "1", "2", "3", "4",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )

    Invoke-Task -Name "prefix branch-topk grid" -ArgList @(
        "20260311/tools/run_prefix_param_grid.py",
        "--suite", "all_known",
        "--exp-name", "exp_prefix_branch_grid",
        "--out-root", $OutRoot,
        "--variants", "rh_mcts_prefix_beam",
        "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
        "--parameter", "PREFIX_BEAM_BRANCH_TOPK",
        "--values", "2", "3", "4", "5",
        "--n-seeds", "1",
        "--seed0", "1000",
        "--solver-time-limit-s", "300"
    )
}

if ($Stage -in @("all", "layout")) {
    Invoke-Task -Name "layout compare industrial large" -ArgList @(
        "20260311/tools/render_layout_compare.py",
        "--run-dir", (Join-Path $OutRootAbs "exp_main_ind_all_1seed_300s"),
        "--case-id", $IndustrialLargeCase,
        "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--seed", "1000",
        "--out", (Join-Path (Join-Path $OutRootAbs "figures") "layout_${IndustrialLargeCase}_compare.png")
    )

    Invoke-Task -Name "layout compare industrial medium" -ArgList @(
        "20260311/tools/render_layout_compare.py",
        "--run-dir", (Join-Path $OutRootAbs "exp_main_ind_all_1seed_300s"),
        "--case-id", $IndustrialMediumCase,
        "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--seed", "1000",
        "--out", (Join-Path (Join-Path $OutRootAbs "figures") "layout_${IndustrialMediumCase}_compare.png")
    )

    Invoke-Task -Name "layout compare public" -ArgList @(
        "20260311/tools/render_layout_compare.py",
        "--run-dir", (Join-Path $OutRootAbs "exp_main_pub_all_1seed_300s"),
        "--case-id", $PublicCase,
        "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
        "--seed", "1000",
        "--out", (Join-Path (Join-Path $OutRootAbs "figures") "layout_${PublicCase}_compare.png")
    )
}

if ($Stage -in @("all", "figures")) {
    Invoke-Task -Name "summary figures" -ArgList @(
        "20260311/tools/render_paper_summary_charts.py",
        "--result-root", $OutRootAbs,
        "--out-dir", (Join-Path $OutRootAbs "figures")
    )
}
