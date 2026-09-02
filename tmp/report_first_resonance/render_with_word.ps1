param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)

$word = $null
$doc = $null
try {
    $pdfParent = Split-Path -Parent $OutputPdf
    New-Item -ItemType Directory -Force -Path $pdfParent | Out-Null
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($InputDocx, $false, $true)
    $doc.ExportAsFixedFormat($OutputPdf, 17)
}
finally {
    if ($null -ne $doc) {
        $doc.Close(0)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
    if ($null -ne $doc) {
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) | Out-Null
    }
    if ($null -ne $word) {
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
