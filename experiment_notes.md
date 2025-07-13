# MODIFIED FAIRIFY EXPERIMENT

## 1. Overview

This document outlines the modifications made to the Fairify experiment (*Fairness Verification of Neural Networks*). The focus is on evaluating model fairness before and after modifications using symbolic verification, counterexample extraction, and retraining. Additional fairness metrics were introduced, and results are visualized and analyzed to assess the impact of these changes.

## 2. Datasets

The following datasets were used:

- **GC** (German Credit)
- **BM** (Bank Marketing)
- **AC** (Adult Census)

Each dataset was used with its predefined sensitive attribute and binary label, as configured in Fairify.

## 3. Model Architecture

The base architecture follows Fairify’s original setup:

- Input: One-hot encoded feature vectors
- Hidden layers: 2 layers with ReLU activation
- Output: 1 neuron with Sigmoid activation
- Loss: Binary Crossentropy
- Optimizer: Adam

No changes were made to the layer structure unless specified per dataset.

## 4. Fairness Metrics

The following metrics were tracked and analyzed:

- **Equal Opportunity Difference (EOD)**
- **Average Odds Difference (AOD)**

Metrics were computed on test sets before and after model retraining.

## 5. Verification and Retraining Process

To run the full verification pipeline, including retraining with counterexamples:

```bash
./reproduce-experiment.sh
