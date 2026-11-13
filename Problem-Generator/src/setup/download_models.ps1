# Downloads the full-size / 8-bit versions of the test models we run on the lab PC (context.md section 4)
# into the project's own Ollama folder, so they can all be deleted later in one go.
# Downloading uses only internet and disk, not the GPU. Safe to rerun: `ollama pull` skips what is
# already downloaded and resumes a broken download.
#
# The lab PC is shared and someone else may already run Ollama on the usual port (11434), so we run our own
# Ollama on port 11500 with our own folder. Start it first, in its own PowerShell window, and leave it open:
#   $env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\LLMxPsychology-models"; ollama serve
# Then, in a second window:
#   powershell -ExecutionPolicy Bypass -File .\download_models.ps1
#
# Tags checked on ollama.com on 2026-11-06. Qwen3.5 has no fp16 tag; bf16 is its full-size version.
# Qwen3.5 27B and Gemma 4 31B are not here: they don't fit the 24 GB GPU at full size (plan: API).

$Folder = "E:\LLMxPsychology-models"
$OurHost = "127.0.0.1:11500"
$env:OLLAMA_HOST = $OurHost   # every ollama command below talks to our Ollama, not anyone else's
$Models = @(
    "llama3.2:3b-instruct-fp16",   # Llama 3.2 3B        full size   6.4 GB
    "qwen3.5:4b-bf16",             # Qwen3.5 4B          full size   9.3 GB
    "olmo-3:7b-instruct-fp16",     # OLMo 3 7B Instruct  full size    15 GB
    "olmo-3:7b-think-fp16",        # OLMo 3 7B Think     full size    15 GB
    "llama3.1:8b-instruct-fp16",   # Llama 3.1 8B        full size    16 GB
    "qwen3.5:9b-bf16",             # Qwen3.5 9B          full size    19 GB
    "gemma4:12b-it-q8_0",          # Gemma 4 12B         8-bit        13 GB
    "qwen3:14b-q8_0"               # Qwen3 14B           8-bit        16 GB
)
$NeededGB = 115   # ~110 GB of models plus a little room

# ---------- checks before downloading ----------

ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "STOP: our Ollama is not running on $OurHost. In another PowerShell window run this and leave it open:"
    Write-Host "  `$env:OLLAMA_HOST=`"$OurHost`"; `$env:OLLAMA_MODELS=`"$Folder`"; ollama serve"
    exit 1
}
$drive = Get-PSDrive -Name $Folder.Substring(0, 1)
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -lt $NeededGB) {
    Write-Host "STOP: only $freeGB GB free on $($drive.Name):, need about $NeededGB GB."
    exit 1
}
Write-Host "Saving to $Folder ($freeGB GB free). $($Models.Count) models, about 110 GB in total."

# ---------- download ----------

$failed = @()
$first = $true
foreach ($m in $Models) {
    Write-Host "`n=== $m ==="
    ollama pull $m
    if ($LASTEXITCODE -ne 0) {
        $failed += $m
        continue
    }
    # After the first download, make sure the running Ollama really saves into our folder (not C:).
    if ($first) {
        $first = $false
        if (-not (Test-Path (Join-Path $Folder "manifests"))) {
            Write-Host "STOP: the download did not land in $Folder."
            Write-Host "Close the Ollama window and start it again with exactly:"
            Write-Host "  `$env:OLLAMA_HOST=`"$OurHost`"; `$env:OLLAMA_MODELS=`"$Folder`"; ollama serve"
            exit 1
        }
    }
}

# ---------- summary ----------

Write-Host "`n=== Done ==="
if ($failed.Count -gt 0) {
    Write-Host "These failed (just rerun the script, it picks up where it stopped):"
    $failed | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "All $($Models.Count) models downloaded."
}
ollama list
