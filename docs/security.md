# Security Notes

This repository should contain framework code only. Do not commit model
weights, datasets, training outputs, server-local scripts, `.env` files, or API
keys.

## W&B

Use W&B credentials through the server environment or `wandb login`:

```bash
export WANDB_API_KEY="<your-key>"
python scripts/train_qwen35_vit.py --wandb --wandb-entity "<entity>"
```

Never place `WANDB_API_KEY`, `wandb.login(key=...)`, tokens, or passwords in
tracked config files or scripts. Public YAML files should keep W&B disabled or
offline by default:

```yaml
logging:
  wandb: false
  wandb_project: vla_adapter_qwen35_vit
  wandb_mode: offline
```

Server-specific helpers belong under ignored paths:

```text
scripts/server/
configs/*.server.yaml
secrets/
```

## Before Publishing

Run these checks before pushing:

```bash
git status --short
git grep -n -I -E "WANDB_API_KEY|wandb\\.login|api[_-]?key|secret|password" -- .
git log --all --oneline -S"WANDB_API_KEY" -- .
git log --all --oneline -S"wandb.login" -- .
```

If a real key was pushed, rotate it immediately. Reverting the commit hides it
from the tip but does not erase it from Git history.
