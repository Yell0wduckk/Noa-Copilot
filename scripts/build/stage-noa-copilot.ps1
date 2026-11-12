param(
    [Parameter(Mandatory=$true)][string]$DistDir,
    [string]$WorktreeRoot = $env:MIACODE_SPINE_SOURCE,
    [string]$PythonEnvSource = $env:MIACODE_NOA_PYTHON_ENV,
    [string]$ModelsSource = $env:MIACODE_SPINE_MODELS,
    [string]$WslDistribution = $env:MIACODE_WSL_DISTRO,
    [string]$WslPython = $env:MIACODE_WSL_PYTHON
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (!$WorktreeRoot) { $WorktreeRoot = Join-Path $repoRoot "../../repos-0wd-v3/worktrees/v15-bms-beat-calibrator" }
if (!$PythonEnvSource) { $PythonEnvSource = Join-Path $repoRoot "third_party/noa-copilot-env" }
if (!$ModelsSource) { $ModelsSource = Join-Path $repoRoot "third_party/spine417-features" }
$runtime = Join-Path $DistDir "noa-copilot"

function Copy-RuntimeTree([string]$Source, [string]$Target) {
    robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed ($LASTEXITCODE): $Source -> $Target" }
}

Copy-RuntimeTree (Join-Path $WorktreeRoot "src/maidata") (Join-Path $runtime "runtime-src/maidata")
Copy-RuntimeTree $PythonEnvSource (Join-Path $runtime "env")
# Only export-format weights belong in the application. Never copy a HF cache.
foreach ($encoder in @("mert", "muq")) {
    $dst = Join-Path $runtime "models/$encoder"
    New-Item -ItemType Directory -Force $dst | Out-Null
    $files = @("config.json","frontend.safetensors")
    if ($encoder -eq "mert") { $files += @("modeling_MERT.py","configuration_MERT.py","preprocessor_config.json") }
    foreach ($name in $files) {
        Copy-Item -LiteralPath (Join-Path $ModelsSource "$encoder/$name") -Destination $dst -Force
    }
}
New-Item -ItemType Directory -Force (Join-Path $runtime "checkpoints") | Out-Null
@{ python=$WslPython; distribution=$WslDistribution } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime "wsl.json")
$revision = git -C $WorktreeRoot rev-parse HEAD
@{ source=$revision; architecture="maidata.spine417.mert6+14-muq8-mel.v4.designer21";
   mert="m-a-p/MERT-v1-330M L6+L14"; muq="OpenMuQ/MuQ-large-msd-iter L8" } |
    ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime "runtime.json")
$global:LASTEXITCODE = 0
