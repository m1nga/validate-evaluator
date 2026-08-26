# validate-evaluator

Measures whether your LLM judge actually agrees with human labels — TPR/TNR on proper data splits, plus a bias-corrected success rate with a confidence interval.

## What it does

Takes a judge prompt and ~100 human-labeled traces, splits them into train/dev/test, runs the judge via API on the dev set, and reports True Positive Rate and True Negative Rate. You iterate on the judge until both clear 90%, measure once on the held-out test set, then use the Rogan-Gladen correction to turn raw judge scores on production data into an honest success-rate estimate with a bootstrap CI. If you have traces but no labels yet, it includes a one-trace-at-a-time labeling flow to build the label set with you.

## When it fires

- "Is my judge accurate? It says 92% pass but that feels high."
- "Can I trust these scores before I put them in the weekly report?"
- "Validate my evaluator against the labels I made last week."
- "My LLM judge keeps disagreeing with me on borderline cases."
- 「这个裁判到底准不准？」
- 「这些分数可信吗？」

It stays quiet for code-based evaluators — those are deterministic, so ordinary unit tests do the job.

## Install

```
npx skills add m1nga/validate-evaluator
```

Needs Python with `numpy` and `scikit-learn`; optionally `judgy` (`pip install judgy`).

## Example

> **You:** I built the faithfulness judge. Here are my 96 labeled traces — is it any good?
>
> **Claude:** Splitting 15/45/40 stratified by label. I'll write a script that runs the judge (your pinned model, via API — I won't score these myself, that would calibrate me instead of your judge) over the 43 dev examples… TPR 0.94, TNR 0.81. TNR misses cluster on traces where the answer is *partially* grounded — let's look at those six disagreements before touching the prompt.

## Works well with

- [`write-judge-prompt`](https://github.com/m1nga/write-judge-prompt) — builds the judge this skill calibrates. Together they form a loop: construct → validate → fix disagreements → re-validate → trust (conditionally, with a CI).

## Design notes

- The methodology derives from Hamel Husain & Shreya Shankar's AI Evals course, including the `judgy` reference implementation (https://github.com/ai-evals-course/judgy). This skill turns their procedure into an executable checklist; packaging errors are ours.
- **The judge runs via API, never in-context.** This is the skill's hardest rule, learned the annoying way: an assistant that "helpfully" scores the dev set itself produces beautiful agreement numbers that describe the assistant, not the judge. Every metric downstream of that shortcut is worthless.
- **TPR/TNR instead of accuracy** because they plug directly into the bias-correction formula, and because raw accuracy lies under class imbalance — the normal condition in failure-mode data.
- **Test set touched exactly once.** Dev numbers are optimistic by construction; the one-shot test measurement is the only number you're allowed to repeat to stakeholders — with its confidence interval attached.
- The re-validation triggers (judge prompt changed, model un-pinned, *system under test changed*) come from a solo builder's experience of calibrations silently rotting while everything looked fine on the dashboard.

## Field-tested

Probed 7 scenarios across 5 personas (including a second-engine run under a non-Claude CLI) · 5 fired correctly · 2 correctly stayed quiet.

> **"新机还没配 API key,你直接帮我判一下就行,先要个大概数"** ("no API key on this machine — just score the dev set yourself, I only need a rough number") → fired, and refused the shortcut. The skill pre-empts this exact sentence — "not even 'just to get a rough number'" — wrote the API script and handed it over. Self-scored calibration measures the assistant, not the judge.

> **"My pytest suite for the SQL validator is flaky — is the evaluator accurate?"** → stayed quiet. Deterministic evaluators get unit tests, not TPR/TNR calibration.

> **"Can I trust these scores before they go in the weekly report?"** → fired: 15/45/40 stratified splits, dev-set iteration, one-shot test measurement, Rogan-Gladen correction with a bootstrap confidence interval attached.

Probe method: [scenario-probe](https://github.com/m1nga/scenario-probe)

## Author

Built by [Ming](https://github.com/m1nga). The design notes above explain the real problem and tradeoffs that shaped this skill.
