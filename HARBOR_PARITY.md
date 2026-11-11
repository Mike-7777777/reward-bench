# RewardBench 2 parity with Harbor

This branch adds what the original-benchmark side of the Harbor adapter's parity
runs needs (harbor-framework/adapters, `src/reward-bench-2`):

- `scripts/run_generative_v2_agent.py`: RewardBench 2 with a coding agent as the
  judge. Data loading, answer positions (seed 0), the four-way prompt, verdict
  parsing and the Ties score are this repository's own functions, used the way
  `scripts/run_generative_v2.py` uses them. The model call is replaced by Claude
  Code in a container, launched the way Harbor's `claude-code` agent launches it,
  with the instruction text of the Harbor task.
- `scripts/harbor_parity_ids.txt`: the 302-task parity subset, a copy of the
  adapter's `parity_task_ids.txt` (all 102 Ties rows and 40 rows from each
  best-of-4 subset).

## Run

Needs Docker and `uv`.

```bash
uv sync
export ANTHROPIC_API_KEY=... CLAUDE_CODE_VERSION=<pinned version>
uv run python scripts/run_generative_v2_agent.py \
  --model <model id> --ids scripts/harbor_parity_ids.txt --num_threads 4
```

Each run writes `rb2_agent_runs/<timestamp>/`: `results.json` (score per subset,
their unweighted mean as `average`, the agent cost, and `agent_errors`),
`per_task.jsonl`, and per-task agent logs under `agent_logs/`. The script exits
with status 1 when an agent run timed out or did not end cleanly (for example an
API error); those tasks are listed in the log and need a rerun before the scores
are used. Ties is left out of the average when the selected rows hold no complete
ref/tied pair.

`--dry_run_dir DIR` writes the instruction of every task as `DIR/<task_id>.md`
without starting an agent. These files are byte-identical to the `instruction.md`
files the Harbor adapter generates.

## Differences from `run_generative_v2.py`

- The judge prompt and the question are sent as one text (the form the script
  uses with `--no_system_prompt`), followed by one sentence that names the output
  file. The agent writes `/workspace/judgment.txt`; its content goes through
  `process_judgement`. A missing or empty file scores 0.
- Ties: the agent rates all responses of a row in one session and writes
  `/workspace/ratings.json`; the ratings go through `process_single_model`.
  Responses are shown in a fixed order derived from sha256, because the dataset
  lists the correct ones first.
