# NER fine-tuned model artifacts

Tokenizer + config for the WangchanBERTa-base NER model fine-tuned on
TMT drug-mention extraction. **Weights (`model.safetensors`) are not
checked into git** because of size.

To run inference:

```
huggingface-cli download <hf-org>/<hf-repo-name> model.safetensors \
  --local-dir experiments/question_understanding/ner_finetuning/artifacts/final_model
```

Then point `NER_MODEL_DIR` (in `apps/api/.env`) at this directory.

If a HuggingFace mirror is not yet published, retrain via
`experiments/question_understanding/ner_finetuning/run/finetune_ner.py`
using the datasets in `../datasets/` and the config saved in
`../results/train_config.json`.
