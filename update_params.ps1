$content = [System.IO.File]::ReadAllText("quant_engine.py")

$content = $content.Replace("tier2_dd = params.get('tier2_dd', -0.08)", "tier2_dd = params.get('tier2_dd', -0.0722)")
$content = $content.Replace("tier1_dd = params.get('tier1_dd', -0.05)", "tier1_dd = params.get('tier1_dd', -0.0385)")
$content = $content.Replace("tp_normal = params.get('tp_normal', 3.0)", "tp_normal = params.get('tp_normal', 3.44)")
$content = $content.Replace("tp_tier1 = params.get('tp_tier1', 2.0)", "tp_tier1 = params.get('tp_tier1', 2.75)")
$content = $content.Replace("sl_mul = params.get('sl_mul', 1.2)", "sl_mul = params.get('sl_mul', 1.43)")
$content = $content.Replace("max_hold_trend = params.get('max_hold_trend', 7)", "max_hold_trend = params.get('max_hold_trend', 15)")
$content = $content.Replace("max_hold_rev = params.get('max_hold_rev', 3)", "max_hold_rev = params.get('max_hold_rev', 4)")

[System.IO.File]::WriteAllText("quant_engine.py", $content)
Write-Output "Updated successfully!"
