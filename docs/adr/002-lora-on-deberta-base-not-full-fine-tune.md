# ADR-002 — LoRA on DeBERTa-v3-base instead of full fine-tuning or -large

**Status:** Accepted (2026-05-23; revisited with v9.0 in PR #3, reverted in PR #8)
**Spec mapping:** §6 "Realizacija na Kaggle (The Deep Learning Component)"

## Context

The spec mandates fine-tuning on Kaggle T4 GPUs (16 GB VRAM each) using
LoRA or QLoRA, with a target F1 above 0.85 on CUAD.

Realistic constraints we measured during v6–v9 iterations:
- A full fine-tune of DeBERTa-v3-base (138 M params) at batch 4 / len
  512 fits one T4 but uses ~14 GB of VRAM in fp16.
- A full fine-tune of DeBERTa-v3-large (439 M) at the same batch
  exhausts a T4 even in fp16 (`v9.0` Kaggle run produced an empty log
  and an empty `failureMessage` — the worker died at model load before
  any user code ran; see PR #3 incident report in PR #8).
- Kaggle has a 9-hour wall clock per kernel; full-FT runs at 2.5 s/step
  on -base ≈ ~3 h, which fits, but checkpoint disk usage doubles.

## Decision

Use **LoRA** on **DeBERTa-v3-base**, with:
- r=64, alpha=128 (v9 — wider than v8's 32/64 to capture more 41-cat boundary)
- target modules: query_proj / key_proj / value_proj
- `pos_weight` per class capped at 50 (v9; was 100 in v8) for inverse
  frequency BCE — addresses CUAD's extreme class imbalance
  (e.g., *Parties* has 4 071 positive examples, *Price Restrictions* has 53)
- per-class threshold tuning on the eval set after training
- `merge_and_unload()` after training and saving the *merged* full model
  for portable inference — adapter-only inference re-initialised the
  pooler head and crashed sigmoid output uniformly to ~0.5; merging is
  what makes the released weights actually usable

## Consequences

**Wins:**
- LoRA adapter on -base has 2.5 M trainable parameters (0.55 % of
  138 M). Training memory is dominated by activations, not by optimizer
  state, so we can use batch 4 / grad_accum 4 (effective batch 16) on
  a single T4.
- Adapter swap is cheap: a future "legal-pretrained" backbone can host
  the same adapter dimensions without retraining the linear classifier head.
- Threshold tuning lifted v7 micro F1 from 0.461 (threshold 0.5) to
  0.671 (per-class tuned) — see [docs/RESULTS.md §1 iteration history](../RESULTS.md).

**Costs (honest):**
- v8 plateaued at micro F1 0.662 / macro F1 0.534 — well below the
  spec's aspirational 0.85. **The 0.85 target is unreachable with the
  current data + model class.** Macro F1 in particular is dragged down
  by four ultra-rare categories (4–7 positives in eval). Closing the
  gap would require: (a) a legal-pretrained backbone (e.g.
  `nlpaueb/legal-bert-base`), (b) more annotated data — CUAD is small
  by ML standards, or (c) a hierarchical / cascade classifier so rare
  categories get their own decision boundary.
- The v9.0 -large attempt failed at model load with no diagnostic; the
  v9.1 retry on -base + tighter weights is what shipped.

## Alternatives considered

1. **Full fine-tuning -base.** Would have given marginal F1 lift over
   LoRA but no thesis-level capability. LoRA's 0.55 % trainable param
   ratio is itself an interesting result — included in the thesis
   methodology chapter.
2. **DeBERTa-v3-large.** First attempt (v9.0) failed with an empty
   Kaggle log; the worker process died at model load. Deferred to a
   future v10 experiment with gradient checkpointing + bf16. **Not
   blocked by code — blocked by Kaggle reliability for the larger
   weight load.**
3. **QLoRA (4-bit).** Considered for v10; rejected for now because the
   GPU we target (T4) does not support all bitsandbytes kernels
   reliably, and the F1 lift from quantisation is in the noise relative
   to model size + data quality.
4. **legal-bert-base.** Most likely to lift F1 toward 0.85 but
   tokeniser-incompatible swap; treated as a v10+ direction.

## References

- E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
  2021.
- [docs/RESULTS.md §1](../RESULTS.md) — full v6 → v8 metric table.
- PR #3 (v9 kernel) + PR #8 (POS_WEIGHT_CAP hotfix) — incident notes
  on the v9.0 silent failure.
