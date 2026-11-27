# Run from a clean project copy containing the archived v2 run.
$ErrorActionPreference = "Stop"
$runtime = Join-Path $env:USERPROFILE "miniconda3/envs/pytorch"
$python = Join-Path $runtime "python.exe"
$env:Path = "$runtime;$runtime/Library/bin;$env:Path"
$env:PYTHONPATH = Join-Path $env:LOCALAPPDATA "baseline-eval-deps"
$env:PYTHONUTF8 = "1"
Push-Location (Resolve-Path (Join-Path $PSScriptRoot "../.."))
try {
    if ((Test-Path runs/gpt_v3_wide) -or (Test-Path runs/gpt_v3_selected) -or (Test-Path runs/gpt_v3_final)) {
        throw "Use a project copy with fresh v3 output directories."
    }
    & $python -u baseline/train.py --architecture residual_unet --base-channels 16 --aux-weight 0.3 --dice-weight 0.5 --long-weight 0.3 --out runs/gpt_v3_wide |
        Tee-Object runs/gpt_v3_wide.log
    if ($LASTEXITCODE) { throw "Training failed" }
    & $python baseline/evaluate.py --checkpoint runs/gpt_v3_wide/best.pt --out runs/gpt_v3_wide/val --split val
    if ($LASTEXITCODE) { throw "Raw validation failed" }
    & $python analysis/GPT-004/select_v3.py
    if ($LASTEXITCODE) { throw "Selection failed" }
    & $python baseline/evaluate.py --checkpoint runs/gpt_v3_selected/best.pt --out runs/gpt_v3_selected/val --split val
    if ($LASTEXITCODE) { throw "Release validation failed" }
    & $python analysis/GPT-004/refine_markers.py
    if ($LASTEXITCODE) { throw "Marker comparison failed" }
    & $python baseline/evaluate.py --checkpoint runs/gpt_v3_final/best.pt --out runs/gpt_v3_final/val --split val
    if ($LASTEXITCODE) { throw "Final checkpoint evaluation failed" }
    & $python analysis/GPT-004/compare_v3.py
    if ($LASTEXITCODE) { throw "Comparison failed" }
    & $python analysis/GPT-004/merge_variant.py
    if ($LASTEXITCODE) { throw "Lower-merge comparison failed" }
}
finally {
    Pop-Location
}
