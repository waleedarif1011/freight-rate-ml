# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space is created by running `scripts/deploy_hf_space.sh` with a Hugging Face **write** token exported as `HF_TOKEN` (or via `huggingface-cli login`). Cursor *app* secrets used for `publish-app` are **not** injected into cloud-agent shells — export the token in your local terminal or CI before running the script. The Space was **not** created from the usability PR.

## One-command deploy (recommended)

```bash
# Token: https://huggingface.co/settings/tokens  (Write + Spaces)
export HF_TOKEN=hf_...
./scripts/deploy_hf_space.sh
```

The script is idempotent:

1. Creates `waleedarif1011/freight-rate-ml` (Gradio SDK) if it does not exist.
2. Uploads Space runtime files via `huggingface_hub.upload_folder` (skips `data/`, tests, docs PDFs).
3. Prints `GET https://huggingface.co/spaces/waleedarif1011/freight-rate-ml` status (expect **200** when the build is up).

Overrides: `HF_USER`, `HF_SPACE`.

## Manual CLI steps

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the write token
huggingface-cli repo create freight-rate-ml --type space --space_sdk gradio -y
# then either re-run ./scripts/deploy_hf_space.sh  OR:
git remote add space https://huggingface.co/spaces/waleedarif1011/freight-rate-ml
git push space main
```

## After deploy

1. Confirm the Space builds and the three presets return rates.
2. Keep the Live Demo badge in `README.md` pointing at `https://huggingface.co/spaces/waleedarif1011/freight-rate-ml` (update only if the slug differs).
3. Set the GitHub repo **Homepage** to that Space URL (maintainer / `gh repo edit`).

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`model_lgbm.txt`, `feature_state.joblib`, `metrics.json`, `lanes.json`)
- `scorer_results/candidate_december.png`
- `assets/` (optional; README screenshots)
