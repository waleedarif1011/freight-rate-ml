# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create freight-rate-ml --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/freight-rate-ml
git push space main
```

3. Confirm the Space builds and the three presets return rates.
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/freight-rate-ml`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`model_lgbm.txt`, `feature_state.joblib`, `metrics.json`, `lanes.json`)
- `scorer_results/candidate_december.png`
