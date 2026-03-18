param(
    [Parameter(Mandatory = $true)]
    [string]$SuiteDir,
    [double]$IntervalS = 60
)

$ErrorActionPreference = 'Stop'

function Append-Log {
    param(
        [string]$Path,
        [string]$Line
    )
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Add-Content -Path $Path -Value $Line
}

$suitePath = (Resolve-Path -LiteralPath $SuiteDir).Path
$statusPath = Join-Path $suitePath 'suite_case_status.csv'
$errPath = Join-Path $suitePath 'launcher.err.log'
$logPath = Join-Path $suitePath 'monitor.log'

Append-Log -Path $logPath -Line ("[{0}] MONITOR_START suite_dir={1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $suitePath)

while ($true) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if (-not (Test-Path $statusPath)) {
        Append-Log -Path $logPath -Line ("[{0}] HEARTBEAT | status_file_missing" -f $ts)
        Start-Sleep -Seconds ([Math]::Max(1, [int][Math]::Round($IntervalS)))
        continue
    }

    try {
        $rows = Import-Csv $statusPath
        $groups = $rows | Group-Object status | Sort-Object Name
        $summary = ($groups | ForEach-Object { "{0}={1}" -f $_.Name, $_.Count }) -join '; '
        $errBytes = if (Test-Path $errPath) { (Get-Item $errPath).Length } else { 0 }
        $py = @(Get-Process python -ErrorAction SilentlyContinue)
        $pyWs = ($py | Measure-Object WS -Sum).Sum
        if ($null -eq $pyWs) { $pyWs = 0 }
        $pyWsGb = [Math]::Round(($pyWs / 1GB), 2)
        Append-Log -Path $logPath -Line ("[{0}] HEARTBEAT | statuses: {1} | py_count={2} | py_ws_gb={3} | err_bytes={4}" -f $ts, $summary, $py.Count, $pyWsGb, $errBytes)
        $active = @($rows | Where-Object { $_.status -in @('pending', 'running') })
        if ($rows.Count -gt 0 -and $active.Count -eq 0) {
            Append-Log -Path $logPath -Line ("[{0}] COMPLETE | statuses: {1} | err_bytes={2}" -f $ts, $summary, $errBytes)
            break
        }
    } catch {
        Append-Log -Path $logPath -Line ("[{0}] HEARTBEAT | status_read_error={1}" -f $ts, $_.Exception.Message)
    }

    Start-Sleep -Seconds ([Math]::Max(1, [int][Math]::Round($IntervalS)))
}
