<#
.SYNOPSIS
    Run `make` for brainchop-c under MSYS2 UCRT64, with MSVC's environment
    loaded only for an explicit legacy CUDA=1 build.

.DESCRIPTION
    Default CUDA, CPU and OpenCL builds need only MSYS2 UCRT64. This helper
    also supports legacy CUDA=1 validation, which needs a second toolchain:

      * nvcc on Windows accepts ONLY MSVC as its host compiler, so cl.exe must
        be on PATH and its INCLUDE/LIB must be set. That is what vcvars64.bat
        does, and the legacy build requires it.

      * MSYS2_PATH_TYPE=inherit is load-bearing. Without it MSYS2 replaces the
        Windows PATH from /etc/profile when the shell starts, and nvcc then
        cannot find cl.exe -- reported as a missing compiler rather than a
        missing PATH.

      * Both toolchains are found by probing rather than hardcoding, so a
        different CUDA version or Build Tools edition needs no edit here.

.EXAMPLE
    scripts\windows_make.ps1 CUDA=1 OPENCL=1 models
    scripts\windows_make.ps1 CUDA=1 OPENCL=1 test-cuda
    scripts\windows_make.ps1 CUDA_ARCH=sm_89 CUDA=1 models   # ~2 s/model, iterating
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$MakeArgs)

$ErrorActionPreference = 'Stop'
$msysBash = 'C:\msys64\usr\bin\bash.exe'
if (-not (Test-Path $msysBash)) { throw "MSYS2 not found at $msysBash. Install it and the UCRT64 toolchain first." }

# The repository's src/ directory, as MSYS2 spells it: /c/src/... rather than C:\src\...
$srcWin = Join-Path (Split-Path -Parent $PSScriptRoot) 'src'
$srcPosix = '/' + ($srcWin -replace ':', '' -replace '\\', '/')
$srcPosix = $srcPosix.Substring(0, 2).ToLower() + $srcPosix.Substring(2)

# Configure MSVC/nvcc only for explicit legacy CUDA=1. Defaults need neither.
if ($MakeArgs -match '^CUDA=1$') {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio Build Tools with the C++ workload -- nvcc requires MSVC as its host compiler." }
    $vsRoot = & $vswhere -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsRoot) { throw "No MSVC C++ toolset found. Install the 'Desktop development with C++' workload in Build Tools." }
    $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat missing under $vsRoot" }

    # Graft the environment vcvars64.bat produces onto this process. Running it
    # in a child cmd and importing `set` is the only supported way -- the batch
    # file cannot export into PowerShell directly.
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] }
    }

    # Newest toolkit wins, sorted numerically: a plain lexical sort puts v9.x
    # after v13.x. Override by putting a different nvcc on PATH before running.
    $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $cuda = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\nvcc.exe') } |
        Sort-Object { [version]($_.Name -replace '^v', '') } | Select-Object -Last 1
    if (-not $cuda) { throw "No CUDA toolkit with nvcc.exe found under $cudaRoot" }
    $env:PATH = (Join-Path $cuda.FullName 'bin') + ';' + $env:PATH
    Write-Host "using $($cuda.Name) and MSVC at $vsRoot" -ForegroundColor DarkGray
}

$env:MSYS2_PATH_TYPE = 'inherit'
$env:MSYSTEM = 'UCRT64'

# Single-quote each argument for the shell. Make variables are routinely passed
# with embedded spaces -- BENCH_ARGS='--backends cpuN1,cpu' is the usual case --
# and joining the array raw splits them back into separate make targets, which
# fails as "No rule to make target 'cpuN1,cpu'". Any literal single quote is
# closed, escaped and reopened, the standard sh idiom.
$quoted = $MakeArgs | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }
& $msysBash -lc "make -C '$srcPosix' $($quoted -join ' ')"
exit $LASTEXITCODE
