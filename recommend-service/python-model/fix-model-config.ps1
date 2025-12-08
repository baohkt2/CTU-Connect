# Script to fix model configuration
# Allows switching between local model and HuggingFace model

Write-Host "=== Model Configuration Fixer ===" -ForegroundColor Green
Write-Host ""
Write-Host "Choose model source:" -ForegroundColor Cyan
Write-Host "1. Use local model (./model/academic_posts_model) - requires config.json" -ForegroundColor White
Write-Host "2. Use HuggingFace PhoBERT (vinai/phobert-base) - downloads automatically" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`nConfiguring for LOCAL model..." -ForegroundColor Cyan
    
    # Check if config.json exists
    $configPath = ".\model\academic_posts_model\config.json"
    if (-not (Test-Path $configPath)) {
        Write-Host "✗ config.json not found!" -ForegroundColor Red
        Write-Host "Creating config.json from PhoBERT template..." -ForegroundColor Yellow
        
        $configContent = @'
{
  "_name_or_path": "vinai/phobert-base",
  "architectures": ["RobertaModel"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 258,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.35.0",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 64001
}
'@
        New-Item -Path (Split-Path $configPath) -ItemType Directory -Force | Out-Null
        Set-Content -Path $configPath -Value $configContent
        Write-Host "✓ Created config.json" -ForegroundColor Green
    } else {
        Write-Host "✓ config.json exists" -ForegroundColor Green
    }
    
    # Update .env
    $envContent = Get-Content ".env" -Raw
    $envContent = $envContent -replace 'MODEL_PATH=.*', 'MODEL_PATH=./model/academic_posts_model'
    Set-Content -Path ".env" -Value $envContent
    Write-Host "✓ Updated .env to use local model" -ForegroundColor Green
    
} elseif ($choice -eq "2") {
    Write-Host "`nConfiguring for HUGGINGFACE model..." -ForegroundColor Cyan
    
    # Update .env
    $envContent = Get-Content ".env" -Raw
    $envContent = $envContent -replace 'MODEL_PATH=.*', 'MODEL_PATH=vinai/phobert-base'
    Set-Content -Path ".env" -Value $envContent
    Write-Host "✓ Updated .env to use HuggingFace model" -ForegroundColor Green
    Write-Host "⚠  Model will be downloaded on first run (~400MB)" -ForegroundColor Yellow
    
} else {
    Write-Host "Invalid choice!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Configuration Complete! ===" -ForegroundColor Green
Write-Host "Run the service with: .\run-dev.ps1" -ForegroundColor White
Write-Host ""
