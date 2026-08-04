param(
  [string]$OutputPath = "data/blog-attention-entropy.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Get-AttentionEntropyRecords {
  param(
    [string]$ManifestPath,
    [int[]]$Layers
  )

  $wanted = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::Ordinal
  )
  foreach ($layer in $Layers) {
    [void]$wanted.Add(
      "attention::transformer.h.$layer.attn::-::attention_entropy_mean::-"
    )
  }

  $records = @{}
  $reader = [System.IO.StreamReader]::new($ManifestPath)
  try {
    $capturing = $false
    $depth = 0
    $builder = $null

    while (($line = $reader.ReadLine()) -ne $null) {
      if (-not $capturing) {
        $match = [regex]::Match($line, '^\s*"id": "([^"]+)"')
        if (-not $match.Success -or -not $wanted.Contains($match.Groups[1].Value)) {
          continue
        }

        $capturing = $true
        $depth = 1
        $builder = [System.Text.StringBuilder]::new()
        [void]$builder.AppendLine("{")
      }

      [void]$builder.AppendLine($line)
      $depth += ([regex]::Matches($line, '\{')).Count
      $depth -= ([regex]::Matches($line, '\}')).Count

      if ($depth -ne 0) { continue }

      $json = $builder.ToString().TrimEnd()
      if ($json.EndsWith(",")) {
        $json = $json.Substring(0, $json.Length - 1)
      }
      $record = $json | ConvertFrom-Json
      $records[[string]$record.layer] = $record
      [void]$wanted.Remove([string]$record.id)
      $capturing = $false
      $builder = $null

      if ($wanted.Count -eq 0) { break }
    }
  }
  finally {
    $reader.Dispose()
  }

  if ($wanted.Count -gt 0) {
    throw "Missing attention entropy records in $ManifestPath"
  }
  return $records
}

function Get-ObservableRecord {
  param(
    [string]$ManifestPath,
    [string]$Id
  )

  $reader = [System.IO.StreamReader]::new($ManifestPath)
  try {
    $capturing = $false
    $depth = 0
    $builder = $null

    while (($line = $reader.ReadLine()) -ne $null) {
      if (-not $capturing) {
        $match = [regex]::Match($line, '^\s*"id": "([^"]+)"')
        if (-not $match.Success -or $match.Groups[1].Value -ne $Id) {
          continue
        }

        $capturing = $true
        $depth = 1
        $builder = [System.Text.StringBuilder]::new()
        [void]$builder.AppendLine("{")
      }

      [void]$builder.AppendLine($line)
      $depth += ([regex]::Matches($line, '\{')).Count
      $depth -= ([regex]::Matches($line, '\}')).Count

      if ($depth -ne 0) { continue }

      $json = $builder.ToString().TrimEnd()
      if ($json.EndsWith(",")) {
        $json = $json.Substring(0, $json.Length - 1)
      }
      return $json | ConvertFrom-Json
    }
  }
  finally {
    $reader.Dispose()
  }

  throw "Missing observable record $Id in $ManifestPath"
}

function Get-RoundedNumbers {
  param([object[]]$Values)
  $result = [System.Collections.Generic.List[double]]::new()
  foreach ($value in $Values) {
    $result.Add([math]::Round([double]$value, 6))
  }
  return $result.ToArray()
}

$runDefinitions = @(
  [pscustomobject]@{
    id = "baseline"
    label = "12-layer baseline"
    layers = 0..11
  },
  [pscustomobject]@{
    id = "6_layers_nanogpt"
    label = "6-layer nanoGPT"
    layers = @(1)
  },
  [pscustomobject]@{
    id = "no_learning_rate_warmup"
    label = "No learning-rate warmup"
    layers = @(1)
  }
)

$output = [ordered]@{
  metric = "attention_entropy_mean"
  recorded_through_step = 100000
  steps = $null
  runs = [ordered]@{}
  conventional_comparison = $null
}

foreach ($run in $runDefinitions) {
  $manifestPath = Join-Path $projectRoot "data/$($run.id)/manifest.json"
  $records = Get-AttentionEntropyRecords -ManifestPath $manifestPath -Layers $run.layers
  $layerValues = [ordered]@{}

  foreach ($layer in $run.layers) {
    $record = $records[[string]$layer]
    $steps = Get-RoundedNumbers -Values $record.series.steps
    if ($null -eq $output.steps) {
      $output.steps = $steps
    }
    elseif (($steps -join ",") -ne ($output.steps -join ",")) {
      throw "Step grid differs for $($run.id), layer $layer"
    }
    $layerValues[[string]$layer] = Get-RoundedNumbers -Values $record.series.values
  }

  $output.runs[$run.id] = [ordered]@{
    label = $run.label
    layers = $layerValues
  }
}

$baselineManifest = Join-Path $projectRoot "data/baseline/manifest.json"
$weightNormRecord = Get-ObservableRecord `
  -ManifestPath $baselineManifest `
  -Id "weight::transformer.h.1.attn.c_attn.weight::-::l2_norm::-"
$output.conventional_comparison = [ordered]@{
  metric = "weight_l2_norm"
  label = "Block 1 QKV projection weight L2 norm"
  steps = Get-RoundedNumbers -Values $weightNormRecord.series.steps
  values = Get-RoundedNumbers -Values $weightNormRecord.series.values
}

$resolvedOutput = Join-Path $projectRoot $OutputPath
$jsonOutput = $output | ConvertTo-Json -Depth 8 -Compress
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($resolvedOutput, $jsonOutput + [Environment]::NewLine, $utf8NoBom)
Write-Output "Wrote $resolvedOutput"
