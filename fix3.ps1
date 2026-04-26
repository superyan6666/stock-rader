$content = [System.IO.File]::ReadAllText("quant_engine.py")

$startStr = "def simulate_ledger_run("
$startIdx = $content.IndexOf($startStr)

$endStr = "f.write(f`"[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\n{mc_report}`")"
$endIdx = $content.IndexOf($endStr, $startIdx) + $endStr.Length

$simBlock = $content.Substring($startIdx, $endIdx - $startIdx)

# But wait, there might be a \r\n after endStr. Let's include it.
if ($content[$endIdx] -eq 13 -and $content[$endIdx+1] -eq 10) {
    $simBlock += "`r`n"
    $endIdx += 2
} elseif ($content[$endIdx] -eq 10) {
    $simBlock += "`n"
    $endIdx += 1
}

$simBlockNew = $simBlock + "            return fitness_score, portfolio_metrics`n    return -9999.0, portfolio_metrics`n`n"

$content = $content.Remove($startIdx, $endIdx - $startIdx)

$insertIdx = $content.IndexOf("def run_backtest_engine(")
$content = $content.Insert($insertIdx, $simBlockNew)

[System.IO.File]::WriteAllText("quant_engine.py", $content)
Write-Output "Surgery successful"
