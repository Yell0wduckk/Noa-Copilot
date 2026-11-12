param(
    [Parameter(Mandatory = $true)]
    [string]$DistDir,
    [string]$WorktreeRoot = "",
    [string]$PythonEnvSource = "",
    [switch]$SkipPythonEnv
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $repoRoot)

if ([string]::IsNullOrWhiteSpace($WorktreeRoot)) {
    $candidateWorktree = Join-Path $workspaceRoot "repos-0wd-v3\worktrees\v11-compact-spec"
    if (Test-Path $candidateWorktree) {
        $WorktreeRoot = $candidateWorktree
    } else {
        $fallbackRepo = Join-Path $workspaceRoot "repos-0wd-v3"
        if (Test-Path $fallbackRepo) {
            $WorktreeRoot = $fallbackRepo
        }
    }
}

$noaRoot = Join-Path $DistDir "noa-copilot"
$runtimeSrcRoot = Join-Path $noaRoot "runtime-src"
$resourcesV2Root = Join-Path $noaRoot "resources\v2"
$checkpointsRoot = Join-Path $noaRoot "checkpoints"
$cacheRoot = Join-Path $noaRoot "cache\audio-features"
$envDstRoot = Join-Path $noaRoot "env"

New-Item -ItemType Directory -Path $runtimeSrcRoot -Force | Out-Null
New-Item -ItemType Directory -Path $resourcesV2Root -Force | Out-Null
New-Item -ItemType Directory -Path $checkpointsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

Write-Host "Staging NoaCopilot runtime into: $noaRoot"

# 1. Stage inference source tree (maidata package)
if (![string]::IsNullOrWhiteSpace($WorktreeRoot) -and (Test-Path (Join-Path $WorktreeRoot "src\maidata"))) {
    $srcMaidata = Join-Path $WorktreeRoot "src\maidata"
    $dstMaidata = Join-Path $runtimeSrcRoot "maidata"
    if (Test-Path $dstMaidata) {
        Remove-Item -Recurse -Force $dstMaidata
    }
    Copy-Item -Path $srcMaidata -Destination $dstMaidata -Recurse -Force
    Get-ChildItem -Path $dstMaidata -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  + Staged inference source from $srcMaidata"
} else {
    Write-Warning "NoaCopilot: worktree source not found at '$WorktreeRoot'; runtime-src left empty."
}

# 2. Stage static inference resources (stats JSONs, path-catalog, slide_data.json)
$slideDataCandidates = @(
    (Join-Path $WorktreeRoot "src\maidata\slide_data.json"),
    (Join-Path $repoRoot "assets\data\slide_data.json"),
    (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\slide_data.json"),
    (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\slide_data.json"),
    (Join-Path $workspaceRoot "repos\maidata-baseline\data\slide_data.json")
)
foreach ($candidate in $slideDataCandidates) {
    if (Test-Path $candidate) {
        Copy-Item $candidate (Join-Path $resourcesV2Root "slide_data.json") -Force
        Write-Host "  + Staged slide_data.json"
        break
    }
}

$statsPairs = @(
    @{ Name = "compactspec-stats.json"; Candidates = @(
        (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\compactspec-stats.json"),
        (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\compactspec-tos-stats.json"),
        (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\compactspec-stats.json"),
        (Join-Path $workspaceRoot "outputs\compactspec-real-smoke\cache\audio\compactspec-stats.json")
    )},
    @{ Name = "compactspec-tos-stats.json"; Candidates = @(
        (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\compactspec-tos-stats.json"),
        (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\compactspec-tos-stats.json")
    )},
    @{ Name = "logspec-ms-stats.json"; Candidates = @(
        (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\logspec-ms-stats.json"),
        (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\logspec-ms-stats.json"),
        (Join-Path $workspaceRoot "outputs\compactspec-real-smoke\cache\audio\logspec-ms-stats.json")
    )},
    @{ Name = "feature-stats.json"; Candidates = @(
        (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\feature-stats.json"),
        (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\feature-stats.json"),
        (Join-Path $workspaceRoot "outputs\compactspec-real-smoke\output\stats.json")
    )}
)

foreach ($pair in $statsPairs) {
    foreach ($candidate in $pair.Candidates) {
        if (Test-Path $candidate) {
            Copy-Item $candidate (Join-Path $resourcesV2Root $pair.Name) -Force
            Write-Host "  + Staged $($pair.Name)"
            break
        }
    }
}

$catalogCandidates = @(
    (Join-Path $workspaceRoot "tools\MiaCode-1.2.0\assets\ai-chart\path-catalog"),
    (Join-Path $workspaceRoot "repos-0wd-v3\resources\v2\path-catalog"),
    (Join-Path $workspaceRoot "outputs\compactspec-real-smoke\cache\geometry\path-catalog")
)
foreach ($catCandidate in $catalogCandidates) {
    if (Test-Path $catCandidate) {
        $dstCat = Join-Path $resourcesV2Root "path-catalog"
        if (Test-Path $dstCat) {
            Remove-Item -Recurse -Force $dstCat
        }
        Copy-Item -Path $catCandidate -Destination $dstCat -Recurse -Force
        Write-Host "  + Staged path-catalog"
        break
    }
}

# 3. Write checkpoints README (weights are intentionally not bundled while iterating)
$readmeLines = @(
    "NoaCopilot Model Checkpoints Directory (checkpoints)",
    "====================================================",
    "",
    "Drop your trained streaming model checkpoints (.pt / .ckpt) into this folder.",
    "MiaCode's NoaCopilot status bar panel and Preferences page will automatically discover them."
)
Set-Content -Path (Join-Path $checkpointsRoot "README.txt") -Value ($readmeLines -join "`r`n") -Encoding UTF8

# 4. Stage portable Python + PyTorch environment when available
if (!$SkipPythonEnv) {
    if ([string]::IsNullOrWhiteSpace($PythonEnvSource)) {
        $envCandidates = @(
            $env:MIACODE_NOA_PYTHON_ENV,
            (Join-Path $repoRoot "third_party\noa-copilot-env")
        )
        foreach ($cand in $envCandidates) {
            if (![string]::IsNullOrWhiteSpace($cand) -and (Test-Path (Join-Path $cand "python.exe"))) {
                $PythonEnvSource = $cand
                break
            }
        }
    }

    if (![string]::IsNullOrWhiteSpace($PythonEnvSource) -and (Test-Path (Join-Path $PythonEnvSource "python.exe"))) {
        Write-Host "  + Staging portable Python+PyTorch environment from: $PythonEnvSource"
        if (Test-Path $envDstRoot) {
            Remove-Item -Recurse -Force $envDstRoot
        }
        Copy-Item -Path $PythonEnvSource -Destination $envDstRoot -Recurse -Force

        # Strip non-runtime directories to reduce package size
        $pruneRelativePaths = @(
            "include",
            "Doc",
            "tcl",
            "Tools",
            "Lib\test",
            "Lib\tkinter",
            "Lib\idlelib",
            "Lib\ensurepip",
            "Lib\site-packages\pip",
            "Lib\site-packages\setuptools",
            "Lib\site-packages\wheel",
            "Lib\site-packages\torch\test",
            "Lib\site-packages\torch\include"
        )
        foreach ($rel in $pruneRelativePaths) {
            $target = Join-Path $envDstRoot $rel
            if (Test-Path $target) {
                Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
            }
        }
        Get-ChildItem -Path $envDstRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        # Ensure python3XX._pth isolates user/system PYTHONPATH and includes ../runtime-src
        $pyDll = Get-ChildItem -Path $envDstRoot -Filter "python3*.dll" |
            Where-Object { $_.Name -match '^python3\d+\.dll$' } |
            Select-Object -First 1
        if ($null -ne $pyDll) {
            $baseName = [System.IO.Path]::GetFileNameWithoutExtension($pyDll.Name)
            $pthPath = Join-Path $envDstRoot "$baseName._pth"
            $pthLines = @(
                "$baseName.zip",
                ".",
                "Lib",
                "Lib\site-packages",
                "DLLs",
                "..\runtime-src",
                "import site"
            )
            Set-Content -Path $pthPath -Value ($pthLines -join "`r`n") -Encoding ASCII
            Write-Host "  + Configured isolated path file: $pthPath"
        }
    } else {
        Write-Host "  i Portable Python env source not specified (pass -PythonEnvSource or set MIACODE_NOA_PYTHON_ENV to bundle env/)."
    }
}
