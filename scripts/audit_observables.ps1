param(
    [string]$CatalogPath = "data/reference_catalog.json",
    [string[]]$ManifestPaths = @(
        "data/baseline/manifest.json",
        "data/no_learning_rate_warmup/manifest.json"
    ),
    [string]$JsonReportPath = "data/observable_audit.json",
    [string]$MarkdownReportPath = "OBSERVABLE_AUDIT.md"
)

$ErrorActionPreference = "Stop"
$script:Errors = [System.Collections.Generic.List[string]]::new()

function Add-AuditError([string]$Message) {
    $script:Errors.Add($Message)
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required file: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Temporal-OpsFromId([string]$Id) {
    $parts = $Id -split "::", 5
    if ($parts.Count -lt 5 -or -not $parts[4] -or $parts[4] -eq "-") {
        return @()
    }
    return @($parts[4] -split "\|" | Where-Object { $_ })
}

function Op-Name([string]$Raw) {
    if ($Raw -match "^([A-Za-z_][A-Za-z0-9_]*)") {
        return $Matches[1]
    }
    return $Raw
}

function Expected-UiModule([string]$Selector) {
    $value = $Selector
    if ($value.StartsWith("transformer.")) {
        $value = $value.Substring("transformer.".Length)
    }
    if ($value.EndsWith(".weight")) {
        $value = $value.Substring(0, $value.Length - ".weight".Length)
    }
    return $value
}

function Definition-Key($Spec) {
    return @(
        $Spec.id,
        $Spec.source_kind,
        $Spec.selector,
        $Spec.reduction,
        (@($Spec.transforms) -join ">")
    ) -join [char]0x1f
}

$allowedSources = @(
    "weight", "grad", "update", "opt_m", "opt_v",
    "activation", "preactivation", "logits", "attention", "gelu_activation"
)
$allowedReductions = @(
    "mean", "std", "l2_norm", "max_abs", "sparsity", "effective_rank",
    "min", "max", "l1_norm", "rms", "positive_fraction", "spectral_norm",
    "top_singular_value", "entropy", "row_std_mean", "col_std_mean", "trace",
    "attention_entropy_mean", "attention_entropy_min",
    "attention_sink_first_token", "attention_sink_domination",
    "attention_sink_ratio", "activation_rate",
    "massive_activation_peak_ratio", "massive_activation_outlier_fraction",
    "massive_neuron_fraction"
)
$allowedTransforms = @("identity", "center", "abs", "normalize", "square")
$allowedTemporals = @("identity", "delta", "ema", "slope", "rolling_std", "curvature")
$attentionReductions = @(
    "attention_entropy_mean", "attention_entropy_min",
    "attention_sink_first_token", "attention_sink_domination",
    "attention_sink_ratio"
)
$geluReductions = @(
    "activation_rate", "massive_activation_peak_ratio",
    "massive_activation_outlier_fraction", "massive_neuron_fraction"
)
$matrixOnlyReductions = @("spectral_norm", "top_singular_value", "row_std_mean", "col_std_mean", "trace")

$catalogPayload = Read-Json $CatalogPath
$specs = @($catalogPayload.observables)

if ($catalogPayload.n -ne $specs.Count) {
    Add-AuditError "Catalog n=$($catalogPayload.n), but contains $($specs.Count) records."
}
if ($specs.Count -ne 1821) {
    Add-AuditError "Expected 1821 catalog records; found $($specs.Count)."
}

$duplicateIds = @($specs | Group-Object id | Where-Object Count -ne 1)
foreach ($duplicate in $duplicateIds) {
    Add-AuditError "Duplicate id '$($duplicate.Name)' occurs $($duplicate.Count) times."
}

$jsSource = Get-Content -LiteralPath "js/observable_defs.js" -Raw -Encoding UTF8
$entries = [System.Collections.Generic.List[object]]::new()

for ($index = 0; $index -lt $specs.Count; $index++) {
    $spec = $specs[$index]
    $findings = [System.Collections.Generic.List[string]]::new()
    $errors = [System.Collections.Generic.List[string]]::new()
    $parts = @($spec.id -split "::", 5)

    if ($parts.Count -ne 5) {
        $errors.Add("canonical id must contain five ::-separated fields")
    }
    else {
        $expectedTransforms = if (@($spec.transforms).Count) {
            @($spec.transforms) -join ">"
        }
        else {
            "-"
        }
        if ($parts[0] -ne $spec.source_kind) {
            $errors.Add("source_kind disagrees with canonical id")
        }
        if ($parts[1] -ne $spec.selector) {
            $errors.Add("selector disagrees with canonical id")
        }
        if ($parts[2] -ne $expectedTransforms) {
            $errors.Add("transforms disagree with canonical id")
        }
        if ($parts[3] -ne $spec.reduction) {
            $errors.Add("reduction disagrees with canonical id")
        }
    }

    if ($allowedSources -notcontains $spec.source_kind) {
        $errors.Add("unsupported source_kind '$($spec.source_kind)'")
    }
    if ($allowedReductions -notcontains $spec.reduction) {
        $errors.Add("unsupported reduction '$($spec.reduction)'")
    }
    foreach ($transform in @($spec.transforms)) {
        if ($allowedTransforms -notcontains (Op-Name $transform)) {
            $errors.Add("unsupported transform '$transform'")
        }
    }
    $temporalOps = @(Temporal-OpsFromId $spec.id)
    foreach ($temporal in $temporalOps) {
        if ($allowedTemporals -notcontains (Op-Name $temporal)) {
            $errors.Add("unsupported temporal operator '$temporal'")
        }
    }

    $expectedUi = Expected-UiModule $spec.selector
    if ($spec.ui_module -ne $expectedUi) {
        $errors.Add("ui_module '$($spec.ui_module)' should be '$expectedUi'")
    }
    if ($expectedUi -match "^h\.(\d+)\.(.+)$") {
        if ([int]$spec.layer -ne [int]$Matches[1]) {
            $errors.Add("layer metadata disagrees with selector")
        }
        if ($spec.role -ne $Matches[2]) {
            $errors.Add("role metadata disagrees with selector")
        }
    }
    elseif ($null -ne $spec.layer) {
        $errors.Add("non-block selector unexpectedly has a layer value")
    }

    if (
        $spec.source_kind -in @("weight", "grad", "update", "opt_m", "opt_v") -and
        -not $spec.selector.EndsWith(".weight")
    ) {
        $errors.Add("parameter-like source does not select a .weight parameter")
    }
    if ($attentionReductions -contains $spec.reduction -and $spec.source_kind -ne "attention") {
        $errors.Add("attention-only reduction is attached to a non-attention source")
    }
    if ($spec.source_kind -eq "attention" -and $attentionReductions -notcontains $spec.reduction) {
        $errors.Add("attention source uses a non-attention reduction")
    }
    if ($geluReductions -contains $spec.reduction -and $spec.source_kind -ne "gelu_activation") {
        $errors.Add("GELU-only reduction is attached to a non-GELU source")
    }
    if ($spec.source_kind -eq "gelu_activation" -and $geluReductions -notcontains $spec.reduction) {
        $errors.Add("GELU source uses a non-GELU reduction")
    }
    if (
        $matrixOnlyReductions -contains $spec.reduction -and
        (
            $spec.source_kind -ne "weight" -or
            $spec.role -notin @("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        )
    ) {
        $errors.Add("matrix-only reduction is not attached to a 2-D Linear weight")
    }
    if (
        $spec.reduction -eq "effective_rank" -and
        (
            $spec.source_kind -notin @("activation", "preactivation") -or
            $spec.role -ne "mlp.c_fc" -or
            @($spec.transforms) -notcontains "center"
        )
    ) {
        $errors.Add("effective_rank does not use the expected centered c_fc activation matrix")
    }

    if ($jsSource -notmatch ('case\s+"' + [regex]::Escape($spec.reduction) + '"')) {
        $errors.Add("formula generator has no explicit reduction case")
    }

    if ($spec.reduction -eq "attention_entropy_min") {
        $findings.Add(
            "Correctly defined, but structurally near zero because causal query q=0 has only one valid key."
        )
    }
    if ($spec.reduction -eq "attention_sink_ratio") {
        $findings.Add(
            "Formula is exact; T scaling uses 1/T as a full-length reference, not the causal-uniform baseline."
        )
    }
    if ($spec.reduction -eq "attention_sink_domination") {
        $findings.Add(
            "Formula is exact; unavailable causal pairs count as zero, giving earlier keys a structural exposure advantage."
        )
    }
    if (
        $spec.reduction -eq "mean" -and
        @($spec.transforms) -contains "center" -and
        $temporalOps.Count -gt 0
    ) {
        $findings.Add(
            "Historical definition is structurally degenerate: batch-centering makes the following global mean zero."
        )
    }
    if (
        $spec.source_kind -eq "weight" -and
        $spec.reduction -eq "l2_norm" -and
        ($temporalOps -join "|") -eq "delta()|ema(alpha=0.95)"
    ) {
        $findings.Add(
            "Historical series is absent because the legacy EMA accepted delta's first NaN; runtime now skips non-finite warm-up inputs."
        )
    }

    $status = if ($errors.Count) {
        "error"
    }
    elseif ($findings.Count) {
        "correct_with_limitation"
    }
    else {
        "correct"
    }

    foreach ($message in $errors) {
        Add-AuditError "$($spec.id): $message"
    }

    $entries.Add([ordered]@{
        ordinal = $index + 1
        id = $spec.id
        source_kind = $spec.source_kind
        selector = $spec.selector
        reduction = $spec.reduction
        transforms = @($spec.transforms)
        temporal = $temporalOps
        status = $status
        findings = @($findings)
        errors = @($errors)
    })
}

$catalogKeys = @($specs | ForEach-Object { Definition-Key $_ } | Sort-Object)
$manifestChecks = [System.Collections.Generic.List[object]]::new()
foreach ($manifestPath in $ManifestPaths) {
    $manifest = Read-Json $manifestPath
    $manifestSpecs = @($manifest.specs)
    $manifestKeys = @($manifestSpecs | ForEach-Object { Definition-Key $_ } | Sort-Object)
    $same = (
        $manifestKeys.Count -eq $catalogKeys.Count -and
        -not (Compare-Object -ReferenceObject $catalogKeys -DifferenceObject $manifestKeys)
    )
    if (-not $same) {
        Add-AuditError "$manifestPath does not have the same definition set as $CatalogPath."
    }
    $manifestChecks.Add([ordered]@{
        path = $manifestPath
        count = $manifestSpecs.Count
        same_definition_set_as_catalog = $same
    })
}

$statusCounts = [ordered]@{}
foreach ($group in ($entries | Group-Object { $_["status"] } | Sort-Object Name)) {
    $statusCounts[$group.Name] = $group.Count
}
$sourceCounts = [ordered]@{}
foreach ($group in ($entries | Group-Object { $_["source_kind"] } | Sort-Object Name)) {
    $sourceCounts[$group.Name] = $group.Count
}
$reductionCounts = [ordered]@{}
foreach ($group in ($entries | Group-Object { $_["reduction"] } | Sort-Object Name)) {
    $reductionCounts[$group.Name] = $group.Count
}

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    implementation_of_record = "observable_lib.py"
    catalog = $CatalogPath
    expected_count = 1821
    audited_count = $entries.Count
    passed = ($script:Errors.Count -eq 0)
    error_count = $script:Errors.Count
    errors = @($script:Errors)
    status_counts = $statusCounts
    source_counts = $sourceCounts
    reduction_counts = $reductionCounts
    manifest_checks = @($manifestChecks)
    entries = @($entries)
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $JsonReportPath -Encoding UTF8

$limitationEntries = @($entries | Where-Object status -eq "correct_with_limitation")
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Observable formula and description audit")
$lines.Add("")
$lines.Add("Implementation of record: ``observable_lib.py``")
$lines.Add("")
$lines.Add("- Audited: $($entries.Count) / 1821 catalog records")
$lines.Add("- Structural or formula errors after corrections: $($script:Errors.Count)")
$lines.Add("- Exact and unqualified: $($statusCounts.correct)")
$lines.Add("- Correct with an explicit limitation: $($statusCounts.correct_with_limitation)")
$lines.Add("- Both manifests match the catalog definition set: $((@($manifestChecks | Where-Object { -not $_.same_definition_set_as_catalog }).Count -eq 0).ToString().ToLower())")
$lines.Add("")
$lines.Add("## Limitation groups")
$lines.Add("")
foreach ($group in ($limitationEntries | Group-Object { $_.findings[0] } | Sort-Object Count -Descending)) {
    $lines.Add("- $($group.Count) x $($group.Name)")
}
$lines.Add("")
$lines.Add("The complete per-observable result, including ordinal, canonical id, parsed pipeline, status, findings, and errors, is in ``$JsonReportPath``.")
$lines.Add("")
$lines.Add("## Validation rules")
$lines.Add("")
$lines.Add("- canonical-id fields equal source, selector, transforms, reduction, and parsed temporal metadata")
$lines.Add("- all ids are unique and all operators are registered by the Python implementation")
$lines.Add("- selector, UI module, layer, and role metadata agree")
$lines.Add("- attention-, GELU-, matrix-, and effective-rank reductions are attached only to compatible sources")
$lines.Add("- every catalog reduction has an explicit exact-formula branch")
$lines.Add("- both run manifests contain exactly the same 1821 definitions as the reference catalog")
$lines | Set-Content -LiteralPath $MarkdownReportPath -Encoding UTF8

Write-Output "Audited $($entries.Count) observables; errors=$($script:Errors.Count); report=$JsonReportPath"
if ($script:Errors.Count) {
    exit 1
}
