<#
Tao moi truong ao .venv va cai DLNet (Windows PowerShell).

  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1            # CPU torch
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gpu       # torch mac dinh cua PyPI (CUDA)
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -NoCheck   # bo buoc tu kiem tra

Sau do:  .venv\Scripts\Activate.ps1   roi   dlnet selfcheck
#>
param([switch]$Gpu, [switch]$NoCheck)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Venv = if ($env:VENV) { $env:VENV } else { ".venv" }
$TorchIndex = if ($Gpu) { "" } elseif ($env:TORCH_INDEX -ne $null) { $env:TORCH_INDEX } else { "https://download.pytorch.org/whl/cpu" }

function Run {
    & $args[0] $args[1..($args.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Lenh that bai ($LASTEXITCODE): $($args -join ' ')" }
}

$check = "import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)"
$py = $null
foreach ($c in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("py", "-3"), @("python"), @("python3"))) {
    if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
    $rest = @($c | Select-Object -Skip 1)
    & $c[0] @rest -c $check 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $c; break }
}
if (-not $py) { throw "Can Python 3.10 - 3.13 (khong tim thay)." }
$rest = @($py | Select-Object -Skip 1)
Write-Host "[env] dung $($py -join ' ') -> $Venv"

Run $py[0] @rest -m venv $Venv
$vpy = Join-Path $Venv "Scripts\python.exe"
Run $vpy -m pip install --quiet --upgrade pip

if ($TorchIndex) {
    Write-Host "[env] cai torch (CPU) tu $TorchIndex"
    try { Run $vpy -m pip install torch --index-url $TorchIndex }
    catch {
        Write-Warning "Khong dung duoc $TorchIndex, thu torch tu PyPI (nang hon)"
        Run $vpy -m pip install torch
    }
} else {
    Write-Host "[env] cai torch tu PyPI"
    Run $vpy -m pip install torch
}
Write-Host "[env] cai DLNet va cac thu vien con lai"
Run $vpy -m pip install -e ".[dev,design]"

if (-not $NoCheck) {
    Write-Host "[env] tu kiem tra"
    Run $vpy -m dlnet selfcheck
}
Write-Host ""
Write-Host "Xong. Kich hoat moi truong:  $Venv\Scripts\Activate.ps1"
Write-Host "Roi chay thu:                dlnet selfcheck"
