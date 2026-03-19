param(
    [ValidateSet("launch", "lane")]
    [string]$Mode = "launch",
    [ValidateSet("A", "B")]
    [string]$Lane = "A",
    [string]$PlanName = ("paper_overnight_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ProjectRoot
$PythonExe = "python"
$PlanRoot = Join-Path $ProjectRoot ("outputs\" + $PlanName)
$MasterLog = Join-Path $PlanRoot "launcher.log"
$MasterErr = Join-Path $PlanRoot "launcher.err.log"

function Write-Log {
    param(
        [string]$Message,
        [string]$Path
    )
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $Path -Value $line -Encoding UTF8
}

function Set-SystemAwake {
    param([bool]$Enable)
    $signature = @"
using System;
using System.Runtime.InteropServices;
public static class SleepUtil {
  [DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    if (-not ("SleepUtil" -as [type])) {
        Add-Type -TypeDefinition $signature | Out-Null
    }
    [uint32]$ES_CONTINUOUS = 2147483648
    [uint32]$ES_SYSTEM_REQUIRED = 1
    [uint32]$ES_AWAYMODE_REQUIRED = 64
    if ($Enable) {
        [SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null
    } else {
        [SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    }
}

function Write-LaneStatus {
    param(
        [string]$LaneName,
        [string]$TaskName,
        [string]$State,
        [string]$DoneFile,
        [double]$ElapsedSec = 0.0
    )
    $laneStatusPath = Join-Path $PlanRoot ("lane_" + $LaneName + "_status.csv")
    $rows = @()
    if (Test-Path $laneStatusPath) {
        $rows = @(Import-Csv -Path $laneStatusPath)
    }
    $match = $rows | Where-Object { $_.lane -eq $LaneName -and $_.task -eq $TaskName } | Select-Object -First 1
    if ($null -eq $match) {
        $obj = [pscustomobject]@{
            lane = $LaneName
            task = $TaskName
            state = $State
            done_file = $DoneFile
            updated_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            elapsed_s = [math]::Round($ElapsedSec, 2)
        }
        $rows += $obj
    } else {
        $match.state = $State
        $match.done_file = $DoneFile
        $match.updated_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        $match.elapsed_s = [math]::Round($ElapsedSec, 2)
    }
    @($rows) | Export-Csv -Path $laneStatusPath -Encoding UTF8 -NoTypeInformation
}

function New-Task {
    param(
        [string]$Name,
        [string]$DoneFile,
        [string[]]$ArgList
    )
    [pscustomobject]@{
        Name = $Name
        DoneFile = $DoneFile
        ArgList = $ArgList
    }
}

function Get-LaneTasks {
    param([string]$LaneName)
    $outRootArg = "outputs/$PlanName"
    $tasks = @()
    if ($LaneName -eq "A") {
        $tasks += New-Task `
            -Name "industrial_main_short" `
            -DoneFile (Join-Path $PlanRoot "main_industrial_xlsx6_1seed_300s\suite_comparison_table.csv") `
            -ArgList @(
                "20260311/run_benchmark_suite.py",
                "--suite", "industrial_xlsx6",
                "--suite-name", "main_industrial_xlsx6_1seed_300s",
                "--out-root", $outRootArg,
                "--variants", "baseline_greedy", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "1",
                "--inner-jobs", "1",
                "--case-jobs", "2",
                "--resume"
            )
        $tasks += New-Task `
            -Name "industrial_ablation_short" `
            -DoneFile (Join-Path $PlanRoot "paper_ablation_industrial_xlsx6_1seed_300s\ablation_wide.csv") `
            -ArgList @(
                "20260311/run_paper_experiments.py", "ablation",
                "--suite", "industrial_xlsx6",
                "--exp-name", "paper_ablation_industrial_xlsx6_1seed_300s",
                "--out-root", $outRootArg,
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "1",
                "--inner-jobs", "1"
            )
        $tasks += New-Task `
            -Name "industrial_main_formal" `
            -DoneFile (Join-Path $PlanRoot "main_industrial_xlsx6_3seed_3600s\suite_comparison_table.csv") `
            -ArgList @(
                "20260311/run_benchmark_suite.py",
                "--suite", "industrial_xlsx6",
                "--suite-name", "main_industrial_xlsx6_3seed_3600s",
                "--out-root", $outRootArg,
                "--variants", "baseline_greedy", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--solver-time-limit-s", "3600",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "3",
                "--inner-jobs", "1",
                "--case-jobs", "2",
                "--resume"
            )
    } else {
        $tasks += New-Task `
            -Name "public_generalization_short" `
            -DoneFile (Join-Path $PlanRoot "generalization_public_1seed_300s\suite_comparison_table.csv") `
            -ArgList @(
                "20260311/run_benchmark_suite.py",
                "--suite", "public_compatible",
                "--suite-name", "generalization_public_1seed_300s",
                "--out-root", $outRootArg,
                "--variants", "baseline_greedy", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "1",
                "--inner-jobs", "1",
                "--case-jobs", "2",
                "--resume"
            )
        $tasks += New-Task `
            -Name "robustness_data3" `
            -DoneFile (Join-Path $PlanRoot "paper_robustness_data3_5seed_900s\robustness_summary.csv") `
            -ArgList @(
                "20260311/run_paper_experiments.py", "robustness",
                "--suite", "industrial_xlsx6",
                "--case-filter", "data3",
                "--exp-name", "paper_robustness_data3_5seed_900s",
                "--out-root", $outRootArg,
                "--solver-time-limit-s", "900",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "5",
                "--inner-jobs", "1",
                "--variants", "rh_mcts_prefix_beam"
            )
        $tasks += New-Task `
            -Name "robustness_data6" `
            -DoneFile (Join-Path $PlanRoot "paper_robustness_data6_5seed_900s\robustness_summary.csv") `
            -ArgList @(
                "20260311/run_paper_experiments.py", "robustness",
                "--suite", "industrial_xlsx6",
                "--case-filter", "data6",
                "--exp-name", "paper_robustness_data6_5seed_900s",
                "--out-root", $outRootArg,
                "--solver-time-limit-s", "900",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "5",
                "--inner-jobs", "1",
                "--variants", "rh_mcts_prefix_beam"
            )
        $tasks += New-Task `
            -Name "sensitivity_data3" `
            -DoneFile (Join-Path $PlanRoot "paper_sensitivity_data3\sensitivity_summary.csv") `
            -ArgList @(
                "20260311/run_paper_experiments.py", "sensitivity",
                "--suite", "industrial_xlsx6",
                "--case-filter", "data3",
                "--exp-name", "paper_sensitivity_data3",
                "--out-root", $outRootArg,
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "3",
                "--inner-jobs", "1",
                "--variants", "rh_mcts_prefix_beam",
                "--mcts-n-sim-values", "1", "2", "3", "5"
            )
        $tasks += New-Task `
            -Name "sensitivity_data6" `
            -DoneFile (Join-Path $PlanRoot "paper_sensitivity_data6\sensitivity_summary.csv") `
            -ArgList @(
                "20260311/run_paper_experiments.py", "sensitivity",
                "--suite", "industrial_xlsx6",
                "--case-filter", "data6",
                "--exp-name", "paper_sensitivity_data6",
                "--out-root", $outRootArg,
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", "2",
                "--seed0", "1000",
                "--n-seeds", "3",
                "--inner-jobs", "1",
                "--variants", "rh_mcts_prefix_beam",
                "--mcts-n-sim-values", "1", "2", "3", "5"
            )
    }
    return $tasks
}

function Run-Lane {
    param([string]$LaneName)
    $laneLog = Join-Path $PlanRoot ("lane_" + $LaneName + ".log")
    $tasks = Get-LaneTasks -LaneName $LaneName
    Write-Log "Lane $LaneName starting with $($tasks.Count) tasks." $laneLog
    Set-Location $RepoRoot
    Set-SystemAwake -Enable $true
    try {
        foreach ($task in $tasks) {
            Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "pending" -DoneFile $task.DoneFile
        }
        foreach ($task in $tasks) {
            if (Test-Path $task.DoneFile) {
                Write-Log "Skipping $($task.Name); done file already exists: $($task.DoneFile)" $laneLog
                Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "skipped" -DoneFile $task.DoneFile
                continue
            }
            if ($DryRun) {
                $taskArgs = @($task.ArgList | ForEach-Object { [string]$_ })
                Write-Log ("[DryRun] python " + ($taskArgs -join " ")) $laneLog
                Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "dry_run" -DoneFile $task.DoneFile
                continue
            }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $taskArgs = @($task.ArgList | ForEach-Object { [string]$_ })
            Write-Log ("Starting $($task.Name): python " + ($taskArgs -join " ")) $laneLog
            Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "running" -DoneFile $task.DoneFile
            & $PythonExe @taskArgs 2>&1 | Tee-Object -FilePath $laneLog -Append
            $exitCode = $LASTEXITCODE
            $sw.Stop()
            if ($exitCode -ne 0) {
                Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "failed" -DoneFile $task.DoneFile -ElapsedSec $sw.Elapsed.TotalSeconds
                throw "Task $($task.Name) failed with exit code $exitCode"
            }
            if (Test-Path $task.DoneFile) {
                Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "completed" -DoneFile $task.DoneFile -ElapsedSec $sw.Elapsed.TotalSeconds
                Write-Log "Completed $($task.Name) in $([math]::Round($sw.Elapsed.TotalMinutes,2)) min." $laneLog
            } else {
                Write-LaneStatus -LaneName $LaneName -TaskName $task.Name -State "missing_done_file" -DoneFile $task.DoneFile -ElapsedSec $sw.Elapsed.TotalSeconds
                throw "Task $($task.Name) finished without expected done file: $($task.DoneFile)"
            }
        }
        Write-Log "Lane $LaneName completed all tasks." $laneLog
    }
    finally {
        Set-SystemAwake -Enable $false
    }
}

if ($Mode -eq "lane") {
    $PlanRoot = Join-Path $ProjectRoot ("outputs\" + $PlanName)
    New-Item -ItemType Directory -Path $PlanRoot -Force | Out-Null
    Run-Lane -LaneName $Lane
    exit 0
}

New-Item -ItemType Directory -Path $PlanRoot -Force | Out-Null
Set-Location $RepoRoot
Write-Log "Launching overnight plan $PlanName from $RepoRoot" $MasterLog
Write-Log "Plan root: $PlanRoot" $MasterLog

$scriptPath = Join-Path $ProjectRoot "run_paper_overnight.ps1"
$laneAOut = Join-Path $PlanRoot "lane_A.stdout.log"
$laneAErr = Join-Path $PlanRoot "lane_A.stderr.log"
$laneBOut = Join-Path $PlanRoot "lane_B.stdout.log"
$laneBErr = Join-Path $PlanRoot "lane_B.stderr.log"

$laneACommand = "& '$scriptPath' -Mode lane -Lane A -PlanName '$PlanName'"
$laneBCommand = "& '$scriptPath' -Mode lane -Lane B -PlanName '$PlanName'"
if ($DryRun) {
    $laneACommand += " -DryRun"
    $laneBCommand += " -DryRun"
}

$procA = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $laneACommand) -PassThru -WindowStyle Hidden -RedirectStandardOutput $laneAOut -RedirectStandardError $laneAErr
$procB = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $laneBCommand) -PassThru -WindowStyle Hidden -RedirectStandardOutput $laneBOut -RedirectStandardError $laneBErr

$manifest = @(
    [pscustomobject]@{
        plan_name = $PlanName
        plan_root = $PlanRoot
        lane = "A"
        pid = $procA.Id
        stdout = $laneAOut
        stderr = $laneAErr
    }
    [pscustomobject]@{
        plan_name = $PlanName
        plan_root = $PlanRoot
        lane = "B"
        pid = $procB.Id
        stdout = $laneBOut
        stderr = $laneBErr
    }
)
$manifest | Export-Csv -Path (Join-Path $PlanRoot "launcher_manifest.csv") -NoTypeInformation -Encoding UTF8
Write-Log "Started lane A pid=$($procA.Id), lane B pid=$($procB.Id)" $MasterLog
Write-Output "Plan started: $PlanName"
Write-Output "Plan root: $PlanRoot"
Write-Output "Lane A pid: $($procA.Id)"
Write-Output "Lane B pid: $($procB.Id)"
