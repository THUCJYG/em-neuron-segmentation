# Run in a fresh project copy containing the frozen baseline and GPT-001 records.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "../..")
$runtime = Join-Path $env:USERPROFILE "miniconda3/envs/pytorch"
$python = Join-Path $runtime "python.exe"
$env:Path = "$runtime;$runtime/Library/bin;$env:Path"
$env:PYTHONPATH = Join-Path $env:LOCALAPPDATA "baseline-eval-deps"
$env:PYTHONUTF8 = "1"

Push-Location $root
try {
    foreach ($variant in @(
        @("gpt_v1_full", "residual_unet", "0.3", "0"),
        @("gpt_v1_residual", "residual_unet", "0", "0"),
        @("gpt_v1_aux", "small_unet", "0.3", "0"),
        @("gpt_v1_dice", "residual_unet", "0.3", "0.5")
    )) {
        $folder = "runs/$($variant[0])"
        if (Test-Path $folder) { throw "Experiment directory already exists: $folder" }
        & $python -u baseline/train.py --architecture $variant[1] --aux-weight $variant[2] --dice-weight $variant[3] --out $folder |
            Tee-Object "$folder.log"
        if ($LASTEXITCODE) { throw "Training failed: $folder" }
        & $python baseline/evaluate.py --checkpoint "$folder/best.pt" --split val --out "$folder/val"
        if ($LASTEXITCODE) { throw "Validation failed: $folder" }
    }
    & $python analysis/GPT-002/compare.py
    if ($LASTEXITCODE) { throw "Comparison failed" }
}
finally {
    Pop-Location
}
