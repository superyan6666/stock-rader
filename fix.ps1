$content = [System.IO.File]::ReadAllText("quant_engine.py")

$sim = [System.IO.File]::ReadAllText("simulate_ledger_run.py")
# simulate_ledger_run.py was saved with Set-Content which uses ANSI or UTF16, let's just extract it properly from the string directly

$startIdx = $content.IndexOf("def simulate_ledger_run(")
$endStr = "f.write(f`"[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\n{mc_report}`")`n"
$endIdx = $content.IndexOf($endStr, $startIdx) + $endStr.Length

$simBlock = $content.Substring($startIdx, $endIdx - $startIdx)

# Create the new block with the return statement
$simBlockNew = $simBlock + "            return fitness_score, portfolio_metrics`n    return -9999.0, portfolio_metrics`n`n"

# Remove the old block
$content = $content.Remove($startIdx, $endIdx - $startIdx)

# Insert it before run_backtest_engine
$insertIdx = $content.IndexOf("def run_backtest_engine(replay_mode: bool = False) -> None:")
$content = $content.Insert($insertIdx, $simBlockNew)

[System.IO.File]::WriteAllText("quant_engine.py", $content)
Write-Output "Done"
