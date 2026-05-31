# 1. Configurazione parametri
$Command = "uv"
$Arguments = "run main.py"

# Bitmask per l'affinità (Core 2 = terzo core logico)
$CpuBitmask = 4 

Write-Host "[*] Avvio del bot HFT..." -ForegroundColor Cyan

# 2. Avvia il processo in background (senza il parametro errato -Priority)
$Process = Start-Process -FilePath $Command -ArgumentList $Arguments -PassThru -NoNewWindow

if ($Process) {
    $ProcId = $Process.Id
    Write-Host "[+] Processo 'uv' avviato con PID: $ProcId" -ForegroundColor Green

    # Aspetta un attimo per dare il tempo al sistema operativo di registrare il processo
    Start-Sleep -Milliseconds 500

    # 3. Imposta sia la Priorità Alta che l'Affinità sulla CPU
    try {
        $ProcessObj = Get-Process -Id $ProcId -ErrorAction Stop
        
        # Impostiamo la priorità via codice
        $ProcessObj.PriorityClass = "High"
        Write-Host "[+] Priorità impostata su: ALTA (High)" -ForegroundColor Green
        
        # Impostiamo l'affinità
        $ProcessObj.ProcessorAffinity = $CpuBitmask
        Write-Host "[+] Affinità impostata su: Core 2 (Bitmask: $CpuBitmask)" -ForegroundColor Green
        
    } catch {
        Write-Host "[-] Errore nella configurazione delle risorse di sistema: $_" -ForegroundColor Red
    }

    # 4. Mantieni la sessione aperta per vedere l'output di uv
    $Process | Wait-Process
} else {
    Write-Host "[-] Fallimento durante l'avvio del comando uv." -ForegroundColor Red
}