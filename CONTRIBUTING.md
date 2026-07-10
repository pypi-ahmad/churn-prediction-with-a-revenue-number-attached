# Contributing

Thanks for your interest in **Churn Prediction with a Revenue Number Attached**. This is a tutorial / portfolio project: three Jupyter notebooks that predict churn, compare classical models with Google TabFM, and attach a revenue figure to at-risk customers.

## Ways to contribute

- **Bug reports** — notebook failures, install issues, broken data URLs, wrong metrics in the README  
- **Documentation** — clearer setup steps, troubleshooting, conceptual explanations  
- **Code / notebooks** — reproducibility fixes, clearer EDA, better tests of assumptions, performance improvements  
- **Ideas** — open an issue first for larger changes (new datasets, different label definitions, CLI wrappers)

## Before you start

1. Read the [README](README.md) (setup, architecture, limitations, licenses).  
2. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).  
3. Search [existing issues](https://github.com/pypi-ahmad/churn-prediction-with-a-revenue-number-attached/issues) to avoid duplicates.

## Development setup

```bash
git clone https://github.com/pypi-ahmad/churn-prediction-with-a-revenue-number-attached.git
cd churn-prediction-with-a-revenue-number-attached

uv python install 3.13.13
uv python pin 3.13.13
uv sync

uv run python -m ipykernel install --user \
  --name churn-revenue-project \
  --display-name "Python (churn-revenue-project)"
```

Prefer the project kernel when running notebooks. A GPU is strongly recommended for TabFM; CPU works but is slower and more memory-sensitive.

### Quick sanity check

```bash
uv run python -c "import tabfm, torch, lazypredict, sklearn; print(tabfm.__version__, torch.cuda.is_available())"
```

### Re-run a notebook (optional, after meaningful changes)

```bash
MPLBACKEND=Agg uv run python notebooks/01_iranian_churn.py
# or full execute:
uv run jupytext --to ipynb notebooks/01_iranian_churn.py -o notebooks/01_iranian_churn.ipynb
MPLBACKEND=Agg uv run jupyter nbconvert --to notebook --execute notebooks/01_iranian_churn.ipynb \
  --output 01_iranian_churn.ipynb \
  --ExecutePreprocessor.kernel_name=churn-revenue-project \
  --ExecutePreprocessor.timeout=3600
```

If you change modeling logic, update **real numbers** in the README only when you have re-run evidence—do not invent metrics.

## Pull request process

1. Fork the repo and create a branch from `main`  
   (`git checkout -b fix/short-description`).  
2. Make focused changes (one concern per PR when possible).  
3. Keep secrets out of the tree (no `.env`, tokens, or private data).  
4. Do not commit `.venv/`, Hugging Face caches, or bulk downloaded datasets.  
5. Open a PR against `main` with:
   - What changed and why  
   - How you verified it (commands run, notebook(s) re-executed, or why re-run was unnecessary)  
   - Any limitations or follow-ups  

### Notebook-specific guidance

| Do | Don’t |
|----|--------|
| Keep train/test stratification and `random_state=42` unless you document a deliberate change | Quietly change RFM cutoffs until metrics “look nice” |
| Fit preprocessors on train only | Fit scalers/encoders on full data |
| Disclose TabFM context subsampling | Silently truncate training rows |
| Flag TabFM weight license (non-commercial) when relevant | Imply MIT covers TabFM weights |
| Prefer PR-AUC / churn F1 under imbalance | Rank models on accuracy alone |

## Reporting bugs

Use the **Bug report** issue template. Include:

- OS and Python version  
- `uv` / package versions if install-related  
- Exact command and full traceback  
- Whether CUDA is available  

## Suggesting features

Use the **Feature request** template. Explain the user problem, not only the solution. For new datasets or label definitions, describe how evaluation and revenue framing should work.

## License and third-party terms

- Contributions to this repository are offered under the project [MIT License](LICENSE).  
- **TabFM model weights** remain under the upstream non-commercial license—do not contribute weight files or claim commercial clearance.  
- Datasets retain UCI / IBM upstream terms; do not commit redistributed bulk data unless licensing is clear and intentional.

## Questions

Open a **Question** issue if something in the README or notebooks is unclear. Prefer issues over private DMs so answers help others.

Thanks for helping keep this project accurate, reproducible, and honest about limitations.
