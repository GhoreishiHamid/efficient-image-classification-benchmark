# One-command finalization of the efficient image classification benchmark.
# Run at the repository root in Windows PowerShell:
# powershell -NoProfile -ExecutionPolicy Bypass -File ".\finish_project.ps1"
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python virtual environment not found. Open the existing project folder (containing .venv).'
}
if (-not (Test-Path -LiteralPath '.git')) {
    throw 'Not a Git repository. Unzip the package into the root of the existing cloned repository.'
}
if (-not (Test-Path -LiteralPath 'src\final_benchmark.py')) {
    throw 'Missing src\final_benchmark.py. Extract the entire ZIP with folder structure preserved.'
}
$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'main') {
    throw "Expected Git branch 'main'; current branch is '$branch'. No files published."
}

function Invoke-PythonStep {
    param([string]$Title, [string[]]$Arguments)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Title failed. Fix the reported error before committing." }
}

Invoke-PythonStep -Title 'Quick smoke test' -Arguments @('src\final_benchmark.py', '--smoke-test')
Invoke-PythonStep -Title 'Final two-model / three-seed benchmark' -Arguments @('src\final_benchmark.py', '--epochs', '3', '--seeds', '42', '123', '2026')
Invoke-PythonStep -Title 'All repository unit tests' -Arguments @('-m', 'unittest', 'discover', '-s', 'tests', '-v')

Write-Host "`n=== Publish verified files ===" -ForegroundColor Cyan
& git add -- 'src/final_benchmark.py' 'tests/test_extended_benchmark.py' 'finish_project.ps1' 'README.md' 'docs/extended_benchmark_protocol.md' 'results/extended/'
if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
& git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & git commit -m 'feat: finalize repeated CPU benchmark and scientific report'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Changes remain available locally.' }
} elseif ($LASTEXITCODE -eq 0) {
    Write-Host 'No new changes to commit.'
} else {
    throw 'Unable to inspect staged changes.'
}
& git push origin main
if ($LASTEXITCODE -ne 0) {
    throw 'git push failed. Your results and commit remain on this laptop; do not force push.'
}
& git status
if ($LASTEXITCODE -ne 0) { throw 'Unable to check git status.' }
Write-Host "`nFINALIZED. Review the new extended results in GitHub README.md." -ForegroundColor Green
