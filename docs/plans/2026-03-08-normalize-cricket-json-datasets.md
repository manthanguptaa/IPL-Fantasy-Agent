# Cricket JSON Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python normalization pipeline that parses the downloaded Cricsheet-style JSON datasets and produces a unified `player_match` training dataset for IPL fantasy modeling.

**Architecture:** Use a small stdlib-only Python package to parse match JSON files into structured per-player per-match rows. Start with a deterministic transformation layer that extracts match metadata, batting/bowling/fielding events, and Dream11-style point labels, then expose a CLI that walks one or more dataset folders and writes a normalized CSV for model training.

**Tech Stack:** Python 3 stdlib, `unittest`, CSV output, Cricsheet JSON data

---

### Task 1: Add parser tests for a single match

**Files:**
- Create: `tests/test_normalize_cricket_json.py`
- Create: `src/ipl_fantasy/normalize_cricket_json.py`

**Step 1: Write the failing test**

Add tests that:
- load one sample IPL JSON match from `ipl_json`
- assert the parser returns one row per player in the match
- assert extracted metadata includes `match_id`, `date`, `competition`, `venue`, `team`, and `opponent`
- assert at least one known player has non-zero batting or bowling output

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: FAIL because parser module and function do not exist yet.

**Step 3: Write minimal implementation**

Create a parser module with:
- a `normalize_match_file(path, source_competition)` function
- event aggregation for batting, bowling, wickets, catches, stumpings, run outs, and maiden overs
- Dream11 point calculation for IPL scoring labels

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: PASS

### Task 2: Add folder-walking normalization test

**Files:**
- Modify: `tests/test_normalize_cricket_json.py`
- Modify: `src/ipl_fantasy/normalize_cricket_json.py`

**Step 1: Write the failing test**

Add a test that:
- points the normalizer at a temporary directory containing a couple of JSON files
- expects a list of normalized player rows across all matches
- verifies rows include a `source_dataset` column

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: FAIL because directory normalization is not implemented yet.

**Step 3: Write minimal implementation**

Add:
- a `normalize_dataset_dir(dataset_dir)` function
- detection of dataset name from folder name
- aggregation across all `.json` files in a directory

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: PASS

### Task 3: Add CLI script for writing the training dataset

**Files:**
- Create: `scripts/build_training_dataset.py`
- Modify: `src/ipl_fantasy/normalize_cricket_json.py`
- Test: `tests/test_normalize_cricket_json.py`

**Step 1: Write the failing test**

Add a test that:
- invokes a script entrypoint helper on a small temporary input
- expects a CSV to be written with normalized rows and stable headers

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: FAIL because CSV writing helper or CLI path does not exist yet.

**Step 3: Write minimal implementation**

Create a script that:
- accepts one or more dataset directories
- normalizes all match files
- writes one consolidated CSV
- uses stable field ordering

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_normalize_cricket_json -v`

Expected: PASS

### Task 4: Verify against real project data

**Files:**
- Use: `ipl_json/`
- Use: `t20s_json/`
- Use: `psl_json/`, `sat_json/`, `sma_json/`, `ilt_json/`, and other league folders as needed

**Step 1: Run the normalizer on one small real dataset**

Run: `python scripts/build_training_dataset.py --dataset ipl_json --limit 3 --output tmp_ipl_training.csv`

Expected: CSV written with per-player per-match rows.

**Step 2: Inspect output**

Check:
- row counts look plausible
- fields are populated
- Dream11 point totals are numeric
- one match produces roughly 22 player rows

**Step 3: Run full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS

### Task 5: Prepare for feature engineering handoff

**Files:**
- Modify: `src/ipl_fantasy/normalize_cricket_json.py`

**Step 1: Ensure output schema includes feature-ready columns**

Required columns:
- identifiers: `match_id`, `match_date`, `competition`, `source_dataset`, `player_name`
- context: `team`, `opponent`, `venue`, `city`, `season`, `toss_winner`, `toss_decision`
- participation: `playing_xi`
- batting: `runs`, `balls_faced`, `fours`, `sixes`, `duck`
- bowling: `balls_bowled`, `overs_bowled`, `maidens`, `runs_conceded`, `wickets`
- fielding: `catches`, `stumpings`, `run_out_direct`, `run_out_assist`
- labels: `dream11_points_total`, `batting_points`, `bowling_points`, `fielding_points`, `other_points`

**Step 2: Verify output**

Run the sample command again and confirm all required columns exist.
