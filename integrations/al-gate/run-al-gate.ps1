$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
& 'C:\Python313\python.exe' -m uvicorn server:create_app --factory `
    --host 127.0.0.1 --port 50002 --no-access-log `
    *>> "$env:USERPROFILE\.al-gate\server.log"
