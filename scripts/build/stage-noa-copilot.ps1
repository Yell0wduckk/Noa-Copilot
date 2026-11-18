param(
    [Parameter(Mandatory=$true)][string]$DistDir,
    [string]$WorktreeRoot = $env:MIACODE_FLAT_SOURCE,
    [string]$SourceRevision = $env:MIACODE_FLAT_REVISION,
    [string]$PythonEnvSource = $env:MIACODE_NOA_PYTHON_ENV,
    [string]$ModelsSource = $env:MIACODE_FLAT_MODELS,
    [string]$WslDistribution = $env:MIACODE_WSL_DISTRO,
    [string]$WslPython = $env:MIACODE_WSL_PYTHON,
    [switch]$SourceOnly
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (!$WorktreeRoot) { $WorktreeRoot = Join-Path $repoRoot "../../repos-0wd-v3/worktrees/v18-beat-mertv2-calibrator" }
if (!$SourceRevision) { $SourceRevision = "HEAD" }
if (!$PythonEnvSource) { $PythonEnvSource = Join-Path $repoRoot "third_party/noa-copilot-env" }
if (!$ModelsSource) { $ModelsSource = Join-Path $repoRoot "third_party/flat320-features" }
$runtime = Join-Path $DistDir "noa-copilot"

function Copy-RuntimeTree([string]$Source, [string]$Target) {
    robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed ($LASTEXITCODE): $Source -> $Target" }
}

# The packager supplies a fresh app directory. Export one committed source tree
# so a concurrent training edit or an untracked local checkpoint cannot enter the bundle.
$revision = git -C $WorktreeRoot rev-parse $SourceRevision
if ($LASTEXITCODE) { throw "Unable to resolve $SourceRevision in $WorktreeRoot" }
$sourceDir = New-Item -ItemType Directory -Force (Join-Path $runtime "runtime-src")
$archive = [IO.Path]::GetTempFileName()
try {
    git -C $WorktreeRoot archive --format=tar "--output=$archive" $revision src/maidata
    if ($LASTEXITCODE) { throw "Unable to archive maidata at $revision" }
    tar -xf $archive -C $sourceDir.FullName --strip-components 1
    if ($LASTEXITCODE) { throw "Unable to unpack the maidata runtime" }
    $legacyRevision = "da8a1228af6c3e750d7a1b759751a3712b8b585e"
    $legacyDir = New-Item -ItemType Directory -Force (Join-Path $sourceDir.FullName "maidata/flat_v184")
    git -C $WorktreeRoot archive --format=tar "--output=$archive" $legacyRevision src/maidata/flat
    if ($LASTEXITCODE) { throw "Unable to archive the v18.4 decoder" }
    tar -xf $archive -C $legacyDir.FullName --strip-components 3
    if ($LASTEXITCODE) { throw "Unable to unpack the v18.4 decoder" }
} finally {
    Remove-Item -LiteralPath $archive -Force
}
$identity = Get-Content -Raw (Join-Path $sourceDir.FullName "maidata/flat/__init__.py")
$architecture = [regex]::Match($identity, '(?m)^ARCHITECTURE = "([^"]+)"').Groups[1].Value
@{ source=$revision; architecture=$architecture; legacy_v184_source=$legacyRevision;
   mert="m-a-p/MERT-v2-FullSong L8"; mel="DualWindowMel-125Hz" } |
    ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime "runtime.json")
if ($SourceOnly) { return }

Copy-RuntimeTree $PythonEnvSource (Join-Path $runtime "env")
$dst = New-Item -ItemType Directory -Force (Join-Path $runtime "models/mert-v2")
foreach ($name in @("config.json", "model.safetensors", "configuration_mert2.py", "modeling_mert2.py",
                    "preprocessor_config.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md")) {
    Copy-Item -LiteralPath (Join-Path $ModelsSource "mert-v2/$name") -Destination $dst.FullName -Force
}
New-Item -ItemType Directory -Force (Join-Path $runtime "checkpoints") | Out-Null
$wslSettings = @{ python=$WslPython; distribution=$WslDistribution }
if ($WslDistribution -eq "Ubuntu-22.04") {
    $wslSettings.previous_python = "/home/dev/.venvs/maimai-structure-v1/songformer/bin/python"
}
$wslSettings | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime "wsl.json")
$global:LASTEXITCODE = 0
