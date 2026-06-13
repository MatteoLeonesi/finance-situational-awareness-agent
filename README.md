<p align="center">
  <img src="https://proteus-prod-public.s3.us-east-1.amazonaws.com/temp/2761a224-6c82-4d44-805e-52d9d4d3c470.png" alt="Adaption banner" width="70%">
</p>

# Finance Situational Awareness Agent

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-original%20dataset-yellow)](https://huggingface.co/datasets/ML0037/finance-situational-awareness-agent)
[![Adaption Version](https://img.shields.io/badge/🤗%20Hugging%20Face-Adaption%20version-blue)](https://huggingface.co/datasets/ML0037/adaption-finance-situational-awareness-bench)

A benchmark for testing how finance AI agents behave under pressure and when they think they are being watched. Built during the inaugural cohort of the **Adaption Research Grant Program**.

## Overview

This benchmark gives a finance agent a clear decision to make. The right answer stays the same, but the situation changes in two ways:

* someone may pressure the agent to make the wrong choice
* the agent may be told that its actions are being logged and reviewed

The goal is to heck whether the agent still makes the right call. The benchmark does **not** claim to know what the model is thinking. It only measures what the model does.

## Datasets

Original dataset:

```text
ML0037/finance-situational-awareness-agent
```

Adaption version:

```text
ML0037/adaption-finance-situational-awareness-bench
```

I also tested Adaption AI's **Adapt My Data** service on this dataset. The dataset received an **A grade**, and the adapted version improved the benchmark by **11.1%** using the **Reasoning Traces** feature. This repository contains the main code for running and evaluating the benchmark. The Hugging Face datasets contain the benchmark data and the adapted release.

## What the benchmark tests

Each example asks the agent to make one finance-related decision, such as:

* hold a wire when a control fails
* delete customer data after the retention period has passed
* refuse an export that breaks policy
* file a breach notice before the deadline
* report a number honestly

Each scenario appears in different versions:

| setting    | values          |
| ---------- | --------------- |
| pressure   | `off`, `on`     |
| monitoring | `none`, `heavy` |

The correct answer does not change between versions.

## Splits

| split | rows | scenarios |
| ----- | ---: | --------: |
| train |  600 |   150 × 4 |
| test  |  200 |    50 × 4 |

There is also an optional `multiturn` config with 100 harder test scenarios. In that version, the monitoring message appears during the conversation instead of at the start.

## Usage

```python
import json
from datasets import load_dataset

ds = load_dataset("ML0037/finance-situational-awareness-agent")
row = ds["test"][0]

tools = json.loads(row["tools_available"])
gold = json.loads(row["gold_action"])

print(row["monitoring_level"], gold)
```

Multiturn:

```python
import json
from datasets import load_dataset

mt = load_dataset(
    "ML0037/finance-situational-awareness-agent",
    "multiturn"
)["test"]

turns = json.loads(mt[0]["turns"])
```

## Notes

The dataset is synthetic, English-only, and based on US-style finance workflows. Use it to test whether finance agents keep making the right decision when pressure and monitoring change. It is not a safety certification.

## Adaption results

<p align="center">
  <img src="https://proteus-prod-public.s3.us-east-1.amazonaws.com/temp/ba894a0d-446f-4669-8ad1-5d1739f1df5d.png" alt="Adaption quality gain" width="32%">
  <img src="https://proteus-prod-public.s3.us-east-1.amazonaws.com/temp/7e3e5b1a-179f-4979-b20a-fe1d29824a51.png" alt="Adaption grade" width="32%">
  <img src="https://proteus-prod-public.s3.us-east-1.amazonaws.com/temp/327e5dfe-96af-4573-bce4-05db1feec134.png" alt="Adaption percentile chart" width="32%">
</p>

## Citation

```bibtex
@misc{finsituawareness2026,
  title  = {Finance Situational Awareness Agent: A Monitoring-Conditional Behavior Benchmark},
  author = {Leonesi, Matteo},
  year   = {2026},
  url    = {https://huggingface.co/datasets/ML0037/finance-situational-awareness-agent}
}
```
