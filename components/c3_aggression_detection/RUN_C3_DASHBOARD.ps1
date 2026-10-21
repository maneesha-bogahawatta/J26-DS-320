param(
    [int]$Port = 8000,
    [string]$StateDirectory = ''
)
$ErrorActionPreference = 'Stop'
$C3DashboardRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$C3DashboardPython = (Resolve-Path -LiteralPath (Join-Path $C3DashboardRoot '..\..\.venv\Scripts\python.exe')).Path
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a local port from 1024 to 65535.' }
Set-Location -LiteralPath $C3DashboardRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$C3DashboardArguments = @('-m', 'src.serve_dashboard', '--port', $Port)
if ($StateDirectory) { $C3DashboardArguments += @('--state-dir', $StateDirectory) }
& $C3DashboardPython @C3DashboardArguments
exit $LASTEXITCODE
