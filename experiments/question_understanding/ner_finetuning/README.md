# NER Fine-Tuning

Canonical home for NER fine-tuning work used by the question-understanding pipeline.

## Goal

Train a Thai NER model for GraphRAG question understanding so the system can extract entities such as:

- `DRUG`
- `BRAND`
- `MANUFACTURER`
- `FORM`
- `STRENGTH`

The standalone fine-tuning flow is:

1. extract entity dictionaries from Neo4j
2. generate BIO-labeled training data
3. fine-tune the token classification model
4. save the model and inference configs under `artifacts/`

## Directory Layout

Recovered from `experiment/ner-wangchanberta`:

- `run/finetune_ner.py`
- `run/generate_ner_data.py`
- `run/extract_entities.py`
- `run/analyze_all_entities.py`
- `run/ner_inference_helper.py`
- `integration_with_app/run_aqt_sanity.py`
- `data/train.json`, `data/validation.json`, `data/test.json`
- `data/dataset_manifest.json`
- `data/entities_*.json`
- `configs/ner_inference_config.json`
- `configs/train_config.reference.json`

Keep in Git:

- run scripts
- configs
- dataset manifests and reproducibility splits
- small source entity lists
- smoke-test scaffolding

Keep out of Git:

- trained model weights
- tokenizer exports
- checkpoint directories
- trainer output directories
- large generated artifacts

## Prerequisites

Before running the pipeline, make sure the following are ready:

1. Python environment with the required libraries installed.
   Typical packages used by these scripts are `torch`, `transformers`, `datasets`, `seqeval`, `pythainlp`, `neo4j`, and `python-dotenv`.
2. Neo4j is up and reachable.
3. Repo-level `.env` contains the Neo4j connection values used by `run/extract_entities.py`.

Expected environment variables:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

If `pythainlp` is not available, dataset generation still works, but it falls back to a regex tokenizer. That is acceptable for smoke runs, but for Thai NER training the preferred setup is to have `pythainlp` installed.

## Quick Start

Run the full flow from the repository root:

```powershell
python experiments/question_understanding/ner_finetuning/run/extract_entities.py
python experiments/question_understanding/ner_finetuning/run/generate_ner_data.py
python experiments/question_understanding/ner_finetuning/run/finetune_ner.py
```

## Step 1: Extract Entity Dictionaries

Script:

- `python experiments/question_understanding/ner_finetuning/run/extract_entities.py`

What it does:

- connects to Neo4j using `.env`
- extracts canonical values for `DRUG`, `BRAND`, `MANUFACTURER`, `FORM`, and `STRENGTH`
- writes them into `data/entities_*.json`

Expected outputs:

- `data/entities_drug.json`
- `data/entities_brand.json`
- `data/entities_manufacturer.json`
- `data/entities_form.json`
- `data/entities_strength.json`

Optional quality check:

```powershell
python experiments/question_understanding/ner_finetuning/run/analyze_all_entities.py
```

This creates a simple report under `artifacts/entity_analysis/entity_report.txt` so you can inspect counts, overlaps, and suspicious values before generating the dataset.

## Step 2: Generate BIO-Labeled Dataset

Script:

- `python experiments/question_understanding/ner_finetuning/run/generate_ner_data.py`

What it does:

- loads entity dictionaries from `data/entities_*.json`
- filters noisy or malformed entity strings
- samples question templates
- fills templates with entity values
- builds character spans for each entity mention
- tokenizes the final text
- converts spans to BIO tags
- runs dataset quality checks
- splits data into train, validation, and test

Default behavior:

- target samples: `10000`
- split ratio: `80/10/10`
- tokenizer: `pythainlp newmm` when available, otherwise regex fallback

Useful command examples:

```powershell
python experiments/question_understanding/ner_finetuning/run/generate_ner_data.py --num-samples 10000
python experiments/question_understanding/ner_finetuning/run/generate_ner_data.py --num-samples 2000
```

Expected outputs:

- `data/train.json`
- `data/validation.json`
- `data/test.json`
- `data/dataset_manifest.json`
- `artifacts/generation_report.json`
- `artifacts/generated_records.jsonl`

Notes:

- `train.json`, `validation.json`, and `test.json` are token-level NER datasets.
- each record contains `id`, `tokens`, and `ner_tags`
- `ner_tags` are BIO labels such as `B-DRUG`, `I-BRAND`, and `O`
- `generated_records.jsonl` is a lightweight artifact for optional integration checks and is not part of the training input

## Step 3: Fine-Tune the Model

Script:

- `python experiments/question_understanding/ner_finetuning/run/finetune_ner.py`

Default model checkpoint:

- `airesearch/wangchanberta-base-att-spm-uncased`

What the training script does:

- reads `label_list` from `data/dataset_manifest.json`
- loads `train`, `validation`, and `test` splits
- normalizes labels into ids
- tokenizes with `is_split_into_words=True`
- aligns labels to subword tokens
- trains a Hugging Face token classification model
- evaluates on the validation split during training
- optionally evaluates on the test split at the end
- saves the final model and configs under `artifacts/`

Default training settings are reflected in `configs/train_config.reference.json`:

- batch size: `4`
- eval batch size: `8`
- grad accumulation: `4`
- epochs: `6`
- max sequence length: `128`
- learning rate: `2e-5`
- warmup ratio: `0.10`
- early stopping patience: `2`

Recommended baseline run:

```powershell
python experiments/question_understanding/ner_finetuning/run/finetune_ner.py
```

Useful variations:

```powershell
python experiments/question_understanding/ner_finetuning/run/finetune_ner.py --num-epochs 8 --batch-size 8
python experiments/question_understanding/ner_finetuning/run/finetune_ner.py --max-train-samples 1000 --max-validation-samples 200 --max-test-samples 200
python experiments/question_understanding/ner_finetuning/run/finetune_ner.py --resume-from-checkpoint experiments/question_understanding/ner_finetuning/artifacts/ner_model_output/checkpoint-1000
```

GPU notes:

- `fp16` defaults to `auto`
- if CUDA is available, mixed precision is enabled automatically
- gradient checkpointing is enabled by default to reduce memory pressure

## Training Outputs

By default, training artifacts are written to:

- `experiments/question_understanding/ner_finetuning/artifacts/ner_model_output`

Important outputs:

- `final_model/`
- `train_config.json`
- `ner_inference_config.json`
- `metrics_test.json`
- `classification_report_test.json`

`final_model/` contains the files needed for inference, while the JSON files capture reproducibility and evaluation metadata.

## Optional App Integration Check

If you want to verify that generated NER-style questions still behave reasonably when passed through the app's AQT layer, run this separately:

```powershell
python experiments/question_understanding/ner_finetuning/integration_with_app/run_aqt_sanity.py
```

This check intentionally imports `src.services.aqt` and is kept outside `run/` so the standalone data-generation and fine-tuning flow stays independent from the main application.

## Suggested End-to-End Workflow

For a fresh standalone run:

1. ensure Neo4j is up and `.env` is correct
2. run `extract_entities.py`
3. optionally inspect the entity report
4. run `generate_ner_data.py`
5. confirm `data/train.json`, `validation.json`, `test.json`, and `dataset_manifest.json` were created
6. run `finetune_ner.py`
7. optionally run `integration_with_app/run_aqt_sanity.py` if you want an app-coupling sanity check
8. keep only lightweight reproducibility assets in Git and leave model outputs inside `artifacts/`

For reruns where the entity lists and dataset are already acceptable:

1. skip extraction
2. skip dataset generation unless templates or entity sources changed
3. rerun only `finetune_ner.py`

## Troubleshooting

Neo4j connection fails:

- check `.env`
- verify Neo4j is running
- confirm the credentials match the local database

Dataset generation produces poor Thai tokenization:

- install `pythainlp`
- rerun `generate_ner_data.py`

Training runs out of memory:

- lower `--batch-size`
- keep gradient checkpointing enabled
- reduce `--max-seq-length`
- reduce `--max-train-samples` for debug runs

Training finishes but test metrics are missing:

- make sure `--no-test-eval` was not passed

## Legacy Path Note

The old path `experiments/name_entity_extraction_benckmarks/` is no longer used by the runtime.
It remains only as a historical marker while the repo migration is being finalized.
