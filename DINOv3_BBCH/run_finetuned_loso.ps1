# Stop on any terminating error (equivalent to set -e)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# YAML-driven fine-tuning -> cache -> temporal LOSO experiments.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location -LiteralPath $ProjectRoot

# Load .env file into environment variables (equivalent to: source "${PROJECT_ROOT}/.env")
$envFile = Join-Path -Path $ProjectRoot -ChildPath '.env'
if (Test-Path -LiteralPath $envFile) {
    Get-Content -LiteralPath $envFile | ForEach-Object {
        $line = $_.Trim()
        # Ignore comments and empty lines
        if ($line -and -not $line.StartsWith('#')) {
            # Strip optional 'export ' prefix if present
            if ($line -match '^\s*export\s+') {
                $line = $line -replace '^\s*export\s+', ''
            }
            $parts = $line -split '=', 2
            if ($parts.Count -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim().Trim('"').Trim("'")
                [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }
    }
}

# Resolve python binary (equivalent to ${PYTHON:-python3})
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python3' }

# Run runner script with forwarded arguments
$runnerScript = Join-Path -Path $ScriptDir -ChildPath 'finetuned_loso_runner.py'
& $python $runnerScript @args
exit $LASTEXITCODE