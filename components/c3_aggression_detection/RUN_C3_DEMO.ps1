param(
    [string]$Video = '',
    [double]$Seconds = 8,
    [string]$OutputDirectory = '',
    [switch]$ResearchAlerts,
    [switch]$Serve,
    [switch]$Interactive,
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$C3Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -LiteralPath $C3Root
$C3Python = (Resolve-Path -LiteralPath '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
if ($Interactive) {
    if ($Serve -or $Video) { throw 'Use -Interactive separately from -Serve or -Video.' }
    if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a local port from 1024 to 65535.' }
    & $C3Python -m src.demo_server --port $Port
    exit $LASTEXITCODE
}
if ($Serve) {
    if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a local port from 1024 to 65535.' }
    Write-Output "Open http://127.0.0.1:$Port in your browser. Press Ctrl+C here to stop the local server."
    & $C3Python -m http.server $Port --bind 127.0.0.1 --directory outputs/final_c3_v2
    exit $LASTEXITCODE
}
if ($Video) {
    if ($Seconds -le 0) { throw 'Seconds must be positive.' }
    $C3Source = (Resolve-Path -LiteralPath $Video).Path
    if (-not $OutputDirectory) {
        $C3Stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fffZ')
        $OutputDirectory = Join-Path $C3Root "outputs\user_run_$C3Stamp"
    }
    $C3Output = [System.IO.Path]::GetFullPath($OutputDirectory)
    if (-not $C3Output.StartsWith($C3Root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputDirectory must remain inside Component 03.'
    }
    $C3Arguments = @('-m', 'src.predict_advanced', '--video', $C3Source, '--seconds', $Seconds, '--out-dir', $C3Output, '--render')
    if ($ResearchAlerts) { $C3Arguments += '--research-alerts' }
    & $C3Python @C3Arguments
    if ($LASTEXITCODE -ne 0) { throw "C3 prediction failed with exit code $LASTEXITCODE" }
    Write-Output "Results saved to $C3Output"
    exit 0
}
Write-Output 'Prepared dashboard: outputs\final_c3_v2\index.html'
Write-Output 'Serve it: .\RUN_C3_DEMO.ps1 -Serve'
Write-Output 'Interactive upload, inference and incident reviews: .\RUN_C3_DEMO.ps1 -Interactive'
Write-Output "Process your video: .\RUN_C3_DEMO.ps1 -Video 'C:\path\video.mp4' -Seconds 8"
Write-Output 'Default output: research scores with quality abstention. Operational alerts are disabled.'
