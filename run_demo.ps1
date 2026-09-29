$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$NlpPython = if ($env:STEP21_NLP_PYTHON) { $env:STEP21_NLP_PYTHON } else { Join-Path $env:USERPROFILE "anaconda3\envs\nlp_thesis\python.exe" }
$UiPython = if ($env:STEP21_STREAMLIT_PYTHON) { $env:STEP21_STREAMLIT_PYTHON } else { Join-Path $env:USERPROFILE "anaconda3\envs\streamlit_env\python.exe" }
$Config = if ($env:STEP21_LOCAL_CONFIG) { $env:STEP21_LOCAL_CONFIG } else { Join-Path $Root "local_config.json" }

foreach ($Required in @($NlpPython, $UiPython, $Config)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required local demo resource not found: $Required" }
}

$env:STEP21_LOCAL_CONFIG = $Config
$env:STEP21_BACKEND_URL = "http://127.0.0.1:8765"
$Runtime = Join-Path $Root "runtime"
New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
$BackendOut = Join-Path $Runtime "backend_stdout.log"
$BackendErr = Join-Path $Runtime "backend_stderr.log"

$Backend = Start-Process -FilePath $NlpPython -ArgumentList @((Join-Path $Root "demo\backend_server.py"), "--host", "127.0.0.1", "--port", "8765") -WorkingDirectory (Join-Path $Root "demo") -WindowStyle Hidden -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr -PassThru
try {
    $Ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $Health = Invoke-RestMethod -Uri "$env:STEP21_BACKEND_URL/health" -TimeoutSec 3
            if ($Health.status -eq "ok") { $Ready = $true; break }
        } catch { }
        if ($Backend.HasExited) { break }
    }
    if (-not $Ready) {
        $Details = if (Test-Path -LiteralPath $BackendErr) { Get-Content -LiteralPath $BackendErr -Raw } else { "No backend error log." }
        throw "The Step 21 backend did not start. $Details"
    }
    Push-Location (Join-Path $Root "demo")
    try {
        & $UiPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
    } finally {
        Pop-Location
    }
} finally {
    if ($Backend -and -not $Backend.HasExited) { Stop-Process -Id $Backend.Id -Force }
}
