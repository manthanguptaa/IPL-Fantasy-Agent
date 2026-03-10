# Baseline Model Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train the first baseline fantasy-point forecasting model on the curated male-only feature dataset and produce metrics plus a reusable serialized model artifact.

**Architecture:** Use a simple scikit-learn pipeline with a deterministic time-based split. Start with one regression model that predicts `dream11_points_total` from pre-match features, evaluate it on a held-out recent time window, and save both metrics and the trained pipeline so later simulation work has a concrete baseline to build on.

**Tech Stack:** Python 3, scikit-learn, joblib, unittest, CSV input/output

---

### Task 1: Add failing tests for time split and trainer outputs

**Files:**
- Create: `tests/test_train_baseline_model.py`
- Create: `src/ipl_fantasy/train_baseline_model.py`

**Step 1: Write the failing test**

Add tests that:
- verify a date-based train/validation split puts older rows in train and newer rows in validation
- verify the trainer returns metrics including `rmse` and `mae`
- verify the feature preparation excludes the target column and preserves row counts

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_baseline_model -v`

Expected: FAIL because the training module does not exist yet.

**Step 3: Write minimal implementation**

Create:
- split helper
- feature/target extraction helper
- baseline training function

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_baseline_model -v`

Expected: PASS

### Task 2: Add CLI script for baseline training

**Files:**
- Create: `scripts/train_baseline_model.py`
- Modify: `src/ipl_fantasy/train_baseline_model.py`
- Test: `tests/test_train_baseline_model.py`

**Step 1: Write the failing test**

Add a test that:
- invokes a CLI helper on a temporary feature CSV
- expects model and metrics artifacts to be written

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_baseline_model -v`

Expected: FAIL because the CLI does not exist yet.

**Step 3: Write minimal implementation**

Create a script that:
- reads the curated feature CSV
- applies a time-based split
- trains the baseline model
- saves metrics JSON and a serialized model file

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_baseline_model -v`

Expected: PASS

### Task 3: Train on the curated dataset

**Files:**
- Use: `tmp/full_player_match_features_curated.csv`

**Step 1: Run the training script**

Run: `python scripts/train_baseline_model.py --input tmp/full_player_match_features_curated.csv --output-dir tmp/baseline_model`

Expected: model and metrics files written successfully.

**Step 2: Inspect metrics**

Check:
- train and validation row counts
- RMSE and MAE are finite
- most important failure modes can be investigated later

**Step 3: Save training configuration**

Ensure the output includes:
- split boundary
- feature columns used
- target column

### Task 4: Verify readiness for next phase

**Files:**
- Use: `tmp/baseline_model/metrics.json`

**Step 1: Confirm the baseline is usable**

The baseline is ready for next phase if:
- training finishes cleanly
- metrics are saved
- the model artifact reloads successfully
- the data split is time-safe

**Step 2: Handoff**

The next work item after this is quantile or uncertainty modeling for Approach B.
