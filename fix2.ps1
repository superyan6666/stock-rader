$content = [System.IO.File]::ReadAllText("quant_engine.py")

$startStr = "def simulate_ledger_run("
$endRegex = "f\.write\(f`"\[AGENT_FITNESS_SCORE\]: \{fitness_score:\.4f\}\\n\{mc_report\}`"\)[ \t\r\n]*"

$match = [regex]::Match($content, "(?s)$startStr.*?$endRegex")

if ($match.Success) {
    $simBlock = $match.Value
    
    $simBlockNew = $simBlock + "            return fitness_score, portfolio_metrics`n    return -9999.0, portfolio_metrics`n`n"
    
    $content = $content.Remove($match.Index, $match.Length)
    
    $insertIdx = $content.IndexOf("def run_backtest_engine(")
    $content = $content.Insert($insertIdx, $simBlockNew)
    
    [System.IO.File]::WriteAllText("quant_engine.py", $content)
    Write-Output "Surgery successful"
} else {
    Write-Output "Match failed"
}
