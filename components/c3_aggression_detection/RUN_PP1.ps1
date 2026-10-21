param([int]$Port = 8002)
$ErrorActionPreference = 'Stop'
# Uses the original shared dashboard with isolated, authentic C3 replay history.
& (Join-Path $PSScriptRoot 'RUN_C3_DASHBOARD.ps1') -Port $Port -StateDirectory 'outputs/pp1_v1/dashboard_state'
exit $LASTEXITCODE
