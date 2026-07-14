$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
# Windows PowerShell 5 wraps native stderr as a PowerShell error, while Uvicorn
# writes normal startup logs there. Start-Process preserves the native streams
# and lets Task Scheduler receive the real Python exit code.
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.mnemo-gate" | Out-Null
$proc = Start-Process -FilePath 'python.exe' -ArgumentList @(
    '-m', 'uvicorn', 'server:create_app', '--factory',
    '--host', '127.0.0.1', '--port', '50002', '--no-access-log'
) -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$env:USERPROFILE\.mnemo-gate\server-out.log" `
    -RedirectStandardError "$env:USERPROFILE\.mnemo-gate\server.log"
exit $proc.ExitCode
