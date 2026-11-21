# Run RewardBench 2 with a coding agent as the judge, for parity runs against
# the Harbor adapter (harbor-framework/adapters, src/reward-bench-2).
#
# Data loading, answer positions, the four-way prompt, verdict parsing and the
# Ties score all come from this repository, the same way scripts/run_generative_v2.py
# uses them. The only replaced piece is the model call: instead of one API request,
# an agent CLI runs in a container with the instruction text of the Harbor task and
# writes its answer to a file.
#
# Examples:
# python scripts/run_generative_v2_agent.py --model claude-sonnet-5-5 --ids scripts/harbor_parity_ids.txt
# python scripts/run_generative_v2_agent.py --dry_run_dir /tmp/rb2-instructions   # no agent, no cost

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Dataset

from rewardbench import load_eval_dataset_multi, process_single_model
from rewardbench.generative_v2 import format_judge_answers, process_judgement
from rewardbench.random_utils import generate_shuffle_positions

# sha256 over (subset, id, prompt, chosen, rejected) of allenai/reward-bench-2 at
# revision 7ff08853, the revision the Harbor adapter pins.
DATA_SHA256 = "5d6cdea569ecc512468396b910670e8efaa29e8af66218218f27558e9690363b"

# The three texts below are copies of the Harbor adapter's task-template files.
RANKING_TRAILER = (
    "\n\nSave your evaluation, the short explanation followed by your final verdict in the format above, "
    "to the file `/workspace/judgment.txt`. Only the contents of that file are graded.\n"
)
TIES_HEADER = """### Task Description
Please act as an impartial judge and evaluate the quality of each of the {n} responses provided by an
AI assistant to the user query displayed below. Rate every response separately, on its own merits.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Be as objective as possible. Rate each response on a scale of 1 to 10. For each rating, only give a whole number between 1 and 10 (inclusive).

[Query]
{query}

"""  # noqa: E501
TIES_TRAILER = (
    "\n\nSave your ratings to the file `/workspace/ratings.json` as one JSON object that maps each response "
    'number to its rating, for example `{{"1": 7, "2": 3}}`. Rate all {n} responses. '
    "Only the contents of that file are graded.\n"
)

DOCKERFILE = """FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl bash ca-certificates nodejs npm procps \\
    && rm -rf /var/lib/apt/lists/*
ARG CLAUDE_CODE_VERSION=
RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh \\
        | bash -s -- ${CLAUDE_CODE_VERSION} \\
    && /root/.local/bin/claude --version
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /workspace
"""

MODEL_ALIAS_VARS = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)

# Claude Code launched the way Harbor's claude-code agent launches it.
RUN_SCRIPT = (
    "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
    "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
    "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
    'printf "%s" "$RB2_INSTRUCTION" | claude --verbose --output-format=stream-json '
    "--permission-mode=bypassPermissions --print 2>&1 | tee /logs/agent/claude-code.txt; "
    'chown -R "$HOST_UID:$HOST_GID" /workspace /logs/agent'
)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def data_digest(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        chosen = [messages[1]["content"] for messages in row["texts_chosen"]]
        rejected = [messages[1]["content"] for messages in row["texts_rejected"]]
        prompt = row["texts_chosen"][0][0]["content"]
        digest.update(json.dumps([row["subset"], row["id"], prompt, chosen, rejected], ensure_ascii=False).encode())
    return digest.hexdigest()


def build_tasks(dataset, seed):
    """One task per row, with the instruction text and what is needed to score it."""
    nonties = dataset.filter(lambda example: example["subset"] != "Ties")
    ties = dataset.filter(lambda example: example["subset"] == "Ties")
    # Positions are drawn over all non-Ties rows, as in run_generative_v2.py, before any id filter.
    positions = generate_shuffle_positions(len(nonties), seed)

    tasks = []
    for row, shuffle_option in zip(nonties, positions):
        prompt = row["texts_chosen"][0][0]["content"]
        answer_a = row["texts_chosen"][0]
        answer_b = row["texts_rejected"][0]
        answer_c = row["texts_rejected"][1]
        answer_d = row["texts_rejected"][2]
        # same swaps as get_judgement in run_generative_v2.py
        if shuffle_option == 0:
            winner_text = "A"
        elif shuffle_option == 1:
            answer_a, answer_b = answer_b, answer_a
            winner_text = "B"
        elif shuffle_option == 2:
            answer_a, answer_c = answer_c, answer_a
            winner_text = "C"
        elif shuffle_option == 3:
            answer_a, answer_d = answer_d, answer_a
            winner_text = "D"
        system_prompt, user_prompt = format_judge_answers(prompt, answer_a, answer_b, answer_c, answer_d)
        tasks.append(
            {
                "task_id": f"{slugify(row['subset'])}-{slugify(row['id'])}",
                "subset": row["subset"],
                "id": row["id"],
                "protocol": "rankings",
                "instruction": f"{system_prompt}\n\n{user_prompt}" + RANKING_TRAILER,
                "winner_text": winner_text,
            }
        )

    for row in ties:
        task_id = f"ties-{slugify(row['id'])}"
        completions = [messages[1]["content"] for messages in row["texts_chosen"] + row["texts_rejected"]]
        # displayed position -> dataset index, as in the Harbor adapter
        order = sorted(
            range(len(completions)), key=lambda index: hashlib.sha256(f"{task_id}:{index}".encode()).hexdigest()
        )
        responses = "\n\n".join(f"[Response {number}]\n{completions[index]}" for number, index in enumerate(order, 1))
        n = len(completions)
        instruction = (
            TIES_HEADER.format(n=n, query=row["texts_chosen"][0][0]["content"]) + responses + TIES_TRAILER.format(n=n)
        )
        tasks.append(
            {
                "task_id": task_id,
                "subset": row["subset"],
                "id": row["id"],
                "protocol": "ratings",
                "instruction": instruction,
                "order": order,
                "num_correct": row["num_correct"],
            }
        )
    return tasks


def score_ranking(task, judgment):
    """Result mapping of get_judgement; a missing or empty file is 0, as in the Harbor verifier."""
    if judgment is None or not judgment.strip():
        return 0.0
    winner = process_judgement(judgment, None)
    if winner == task["winner_text"]:
        return 1
    elif winner in "ABCD":
        return 0
    return 0.25  # "error": effectively a tie


def parse_ratings(task, raw):
    """Ratings in dataset order; anything but an integer 1-10 is -1, as in get_single_rating."""
    scores = [-1] * len(task["order"])
    try:
        data = json.loads(raw) if raw is not None else None
    except ValueError:
        data = None
    if isinstance(data, dict):
        for position, index in enumerate(task["order"]):
            value = data.get(str(position + 1))
            if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10:
                scores[index] = value
    return scores


class ClaudeCodeRunner:
    """One container per task: python:3.13-slim (the Harbor task image) plus Claude Code."""

    def __init__(self, model, version, timeout_sec, log_dir):
        self.model = model
        self.timeout_sec = timeout_sec
        self.log_dir = Path(log_dir)
        self.image = f"rb2-claude-code:{version or 'latest'}"
        assert os.environ.get("ANTHROPIC_API_KEY"), "Please set ANTHROPIC_API_KEY"
        with tempfile.TemporaryDirectory() as context:
            Path(context, "Dockerfile").write_text(DOCKERFILE)
            subprocess.run(
                ["docker", "build", "-q", "-t", self.image]
                + ["--build-arg", f"CLAUDE_CODE_VERSION={version or ''}", context],
                check=True,
            )

    def __call__(self, task):
        trial = self.log_dir / task["task_id"]
        workspace, logs = trial / "workspace", trial / "agent"
        workspace.mkdir(parents=True)
        logs.mkdir()
        env = {
            "RB2_INSTRUCTION": task["instruction"],
            "ANTHROPIC_MODEL": self.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            "CLAUDE_CONFIG_DIR": "/logs/agent/sessions",
            "HOST_UID": str(os.getuid()),
            "HOST_GID": str(os.getgid()),
        }
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        if "ANTHROPIC_BASE_URL" in env:
            # Harbor's claude-code agent points every model alias at the requested
            # model when a custom endpoint is set, so background calls use it too.
            for name in MODEL_ALIAS_VARS:
                env[name] = os.environ.get(name, self.model)
        command = ["docker", "run", "--rm", "-v", f"{workspace.resolve()}:/workspace"]
        command += ["-v", f"{logs.resolve()}:/logs/agent"]
        for name in env:
            command += ["-e", name]
        command += [self.image, "bash", "-c", RUN_SCRIPT]
        status = "ok"
        try:
            subprocess.run(command, env={**os.environ, **env}, capture_output=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            status = "timeout"
        output = workspace / ("judgment.txt" if task["protocol"] == "rankings" else "ratings.json")
        text = output.read_text(encoding="utf-8", errors="replace") if output.is_file() else None
        return text, {"status": status, **self._final_result(logs / "claude-code.txt")}

    @staticmethod
    def _final_result(stream):
        if not stream.is_file():
            return {}
        for line in reversed(stream.read_text(errors="replace").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                return {key: event.get(key) for key in ("total_cost_usd", "num_turns", "is_error")}
        return {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model id passed to the agent")
    parser.add_argument("--agent", type=str, default="claude-code", choices=["claude-code"])
    parser.add_argument("--agent_version", type=str, default=os.environ.get("CLAUDE_CODE_VERSION"))
    parser.add_argument("--dataset", type=str, default="allenai/reward-bench-2", help="path to huggingface dataset")
    parser.add_argument("--ids", type=str, default=None, help="file with the task ids to run (default: all rows)")
    parser.add_argument("--seed", type=int, default=0, help="seed for deterministic answer-position shuffling")
    parser.add_argument("--num_threads", type=int, default=4, help="number of agent containers run in parallel")
    parser.add_argument("--timeout_sec", type=float, default=600.0, help="agent time limit per task")
    parser.add_argument("--out", type=str, default=None, help="run directory (default: rb2_agent_runs/<timestamp>)")
    parser.add_argument("--dry_run_dir", type=str, default=None, help="only write <task_id>.md instructions here")
    return parser.parse_args()


def main():
    args = get_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.setLevel(logging.INFO)

    # same call as scripts/run_generative_v2.py
    dataset = load_eval_dataset_multi(
        core_set=True,
        dataset=args.dataset,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=None,
        logger=logger,
        keep_columns=["texts_chosen", "texts_rejected", "id", "subset", "num_correct"],
        max_turns=4,
    )
    if args.dataset == "allenai/reward-bench-2":
        digest = data_digest(dataset)
        assert digest == DATA_SHA256, f"dataset content differs from the pinned revision: {digest}"

    tasks = build_tasks(dataset, args.seed)
    if args.ids:
        wanted = set(Path(args.ids).read_text().split())
        unknown = wanted - {task["task_id"] for task in tasks}
        assert not unknown, f"unknown task ids: {sorted(unknown)[:5]}"
        tasks = [task for task in tasks if task["task_id"] in wanted]
    logger.info(f"{len(tasks)} tasks")

    if args.dry_run_dir:
        target = Path(args.dry_run_dir)
        target.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            (target / f"{task['task_id']}.md").write_text(task["instruction"], encoding="utf-8", newline="")
        logger.info(f"wrote {len(tasks)} instructions to {target}")
        return

    assert args.model, "--model is required unless --dry_run_dir is set"
    out = Path(args.out or f"rb2_agent_runs/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out.mkdir(parents=True, exist_ok=True)
    runner = ClaudeCodeRunner(args.model, args.agent_version, args.timeout_sec, out / "agent_logs")

    def run_task(task):
        text, meta = runner(task)
        record = {key: task[key] for key in ("task_id", "subset", "id", "protocol")}
        record["agent"] = meta
        if task["protocol"] == "rankings":
            record["results"] = score_ranking(task, text)
        else:
            record["scores"] = parse_ratings(task, text)
            record["num_correct"] = task["num_correct"]
        return record

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        records = list(executor.map(run_task, tasks))

    results_grouped = {"model": args.model, "agent": f"{args.agent}@{args.agent_version}", "seed": args.seed}
    ties_records = [record for record in records if record["protocol"] == "ratings"]
    for subset in sorted({record["subset"] for record in records}):
        if subset == "Ties":
            ties_dataset = Dataset.from_list(
                [{key: record[key] for key in ("id", "num_correct", "scores")} for record in ties_records]
            )
            _, ties_score = process_single_model(ties_dataset)
            if math.isnan(ties_score):
                # no complete ref/tied pair among the selected rows: left out of the average
                logger.warning("Ties: no complete ref/tied pair, subset skipped")
                continue
            results_grouped[subset] = ties_score
        else:
            results = [record["results"] for record in records if record["subset"] == subset]
            results_grouped[subset] = sum(results) / len(results)
            print(f"{subset}: {sum(results)}/{len(results)} ({results_grouped[subset]})")
    subsets = [key for key in results_grouped if key not in ("model", "agent", "seed")]
    results_grouped["average"] = sum(results_grouped[key] for key in subsets) / len(subsets)
    # a timeout or an agent run that did not end cleanly (API error, crash) is not a judgment
    agent_errors = [
        record["task_id"]
        for record in records
        if record["agent"].get("status") != "ok" or record["agent"].get("is_error") is not False
    ]
    results_grouped["agent_errors"] = len(agent_errors)
    costs = [record["agent"].get("total_cost_usd") or 0 for record in records]
    results_grouped["agent_cost_usd"] = sum(costs)
    print(json.dumps(results_grouped, indent=2))
    (out / "results.json").write_text(json.dumps(results_grouped, indent=2) + "\n")
    (out / "per_task.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    logger.info(f"run directory: {out}")
    if agent_errors:
        logger.error(f"{len(agent_errors)} agent runs failed and need a rerun: {agent_errors[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
