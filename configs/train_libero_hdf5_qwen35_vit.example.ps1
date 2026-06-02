$ErrorActionPreference = "Stop"

$DemoRoot = "data\libero"
$StatsPath = "outputs\libero_action_stats.json"
$OutputDir = "outputs\qwen35_vit_libero"

.\.conda\python.exe scripts\prepare_libero_hdf5.py `
  --root $DemoRoot `
  --output-json $StatsPath `
  --tokenizer-path pretrained_models\Qwen3.5-2B `
  --sample-check

.\.conda\python.exe scripts\train_qwen35_vit.py `
  --libero-hdf5-root $DemoRoot `
  --libero-val-ratio 0.02 `
  --action-stats-json $StatsPath `
  --qwen-path pretrained_models\Qwen3.5-2B `
  --vision-pretrained `
  --vision-cache-dir pretrained_models\vision_cache\hf `
  --use-lora `
  --lora-target language_model `
  --lora-rank 64 `
  --batch-size 8 `
  --grad-accumulation-steps 8 `
  --max-steps 100000 `
  --save-every-steps 10000 `
  --output-dir $OutputDir
