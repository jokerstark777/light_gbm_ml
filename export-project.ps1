<#
.SYNOPSIS
    Merges all Python sources and common config files into one text file.

.DESCRIPTION
    Walks the project root (script directory by default), skipping venv, .git,
    __pycache__, and tool cache folders. Output includes:
    - all *.py files;
    - known config filenames (requirements, pyproject, docker-compose, etc.).

.PARAMETER OutputPath
    Output .txt path (default: project_export.txt in project root).

.PARAMETER ProjectRoot
    Project root folder (default: directory containing this script).
#>
param(
    [string]$OutputPath = "",
    [string]$ProjectRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project directory not found: $ProjectRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot "project_export.txt"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot $OutputPath
}

$ExcludeDirs = @(
    ".git", "venv", ".venv", "env", "ENV", "__pycache__",
    "node_modules", ".cursor", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".eggs", ".tox", ".nox", ".hypothesis"
)

$ConfigFileNames = @(
    "requirements.txt", "requirements-dev.txt", "constraints.txt",
    "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini",
    "mypy.ini", "ruff.toml", ".flake8", "Pipfile", "Pipfile.lock", "poetry.lock",
    ".env.example", "environment.yml", "conda.yml",
    "docker-compose.yml", "docker-compose.yaml", "Dockerfile"
)

function Test-ExcludedPath {
    param([string]$FullPath)
    $norm = $FullPath.Replace([IO.Path]::AltDirectorySeparatorChar, [IO.Path]::DirectorySeparatorChar)
    $segments = $norm.Split([IO.Path]::DirectorySeparatorChar)
    foreach ($d in $ExcludeDirs) {
        if ($segments -contains $d) { return $true }
    }
    foreach ($seg in $segments) {
        if ($seg -like "*.egg-info") { return $true }
    }
    return $false
}

$pyFiles = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Filter "*.py" -ErrorAction SilentlyContinue |
    Where-Object { -not (Test-ExcludedPath $_.FullName) } |
    Sort-Object FullName

$cfgFiles = @()
foreach ($name in $ConfigFileNames) {
    $cfgFiles += Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $name -and -not (Test-ExcludedPath $_.FullName) }
}

$merged = @($pyFiles) + @($cfgFiles)
$unique = $merged | Group-Object FullName | ForEach-Object { $_.Group[0] } | Sort-Object FullName

$bannerLine = "#" * 88
$ruleLine   = "-" * 88
$sb = [System.Text.StringBuilder]::new()
$utf8 = [System.Text.UTF8Encoding]::new($false)
$rootPrefix = $ProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

foreach ($file in $unique) {
    $full = $file.FullName
    $rel = if ($full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        $full.Substring($rootPrefix.Length)
    } else {
        $file.Name
    }
    $relPosix = $rel.Replace([IO.Path]::AltDirectorySeparatorChar, "/")

    [void]$sb.AppendLine($bannerLine)
    [void]$sb.AppendLine("## FILE START")
    [void]$sb.AppendLine("## Path (relative):  $relPosix")
    [void]$sb.AppendLine("## Path (absolute): $full")
    [void]$sb.AppendLine($bannerLine)
    [void]$sb.AppendLine($ruleLine)
    [void]$sb.AppendLine([System.IO.File]::ReadAllText($full, $utf8))
    [void]$sb.AppendLine($ruleLine)
    [void]$sb.AppendLine("## FILE END: $relPosix")
    [void]$sb.AppendLine($bannerLine)
    [void]$sb.AppendLine()
}

[System.IO.File]::WriteAllText($OutputPath, $sb.ToString(), $utf8)
Write-Host "Export written: $OutputPath"
Write-Host "Files included: $($unique.Count)"
