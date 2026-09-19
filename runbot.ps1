# Overnight data collection / paper trading launcher (Windows).
#
#   .\runbot.ps1                  # maker-only paper trading, records ticks + trade prints
#   .\runbot.ps1 -Term            # same, with the Rich terminal dashboard
#   .\runbot.ps1 -Strategy legacy_merton
#
# --strategy is always passed so the process never blocks on the interactive
# strategy prompt: a run left overnight must not sit waiting for a keypress.
# The orchestrator keeps Windows awake on its own (SetThreadExecutionState every 30 s),
# so the screen can go off. Logs land in logs\<date>\merton\live_<ts>\.

param(
    [string]$Strategy = "merton",
    [switch]$Term,
    [int]$CpuBitmask = 0   # 0 = let Windows schedule it; e.g. 4 pins to logical core 2
)

$argList = @("run", "python", "main.py", "--strategy", $Strategy)
if ($Term) { $argList += "--term" }

Write-Host "[*] Avvio bot (strategy: $Strategy)..." -ForegroundColor Cyan

$Process = Start-Process -FilePath "uv" -ArgumentList $argList -PassThru -NoNewWindow
if (-not $Process) {
    Write-Host "[-] Avvio fallito: 'uv' non trovato nel PATH?" -ForegroundColor Red
    exit 1
}

Write-Host "[+] PID: $($Process.Id)" -ForegroundColor Green
Start-Sleep -Milliseconds 500

try {
    $ProcessObj = Get-Process -Id $Process.Id -ErrorAction Stop
    $ProcessObj.PriorityClass = "High"
    Write-Host "[+] Priorita': ALTA" -ForegroundColor Green
    if ($CpuBitmask -gt 0) {
        $ProcessObj.ProcessorAffinity = $CpuBitmask
        Write-Host "[+] Affinita' CPU: $CpuBitmask" -ForegroundColor Green
    }
} catch {
    Write-Host "[-] Impossibile impostare priorita'/affinita': $_" -ForegroundColor Yellow
}

Write-Host "[*] In esecuzione. CTRL+C per fermare." -ForegroundColor Cyan
$Process | Wait-Process
