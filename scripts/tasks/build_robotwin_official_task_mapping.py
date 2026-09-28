#!/usr/bin/env python3
"""Build a mapping from FastWAM RoboTwin episode ids to official RoboTwin task categories.

The script matches:
    data/robotwin2.0/robotwin2.0/meta/episodes.jsonl
against official raw instruction files under:
    data/robotwin2_official/dataset/<task_name>/<subset>/instructions/episode*.json

Matching logic:
1. Instruction containment: all FastWAM instructions must be contained in the
   official episode's `seen` instruction set.
2. Source-order alignment: FastWAM appears to import official RoboTwin episodes
   task-by-task, with each task occupying a contiguous block and raw episode
   files consumed in lexicographic filename order.
3. Length equality: the official video frame count minus 1 must equal
   FastWAM's `length` for the matched episode.

If an official episode satisfies all three constraints, the mapping is treated as
100% confidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mapping from FastWAM RoboTwin episode ids to official RoboTwin "
            "task categories using instruction containment, source order, and "
            "trajectory length agreement."
        )
    )
    parser.add_argument(
        "--processed-meta",
        default="./data/robotwin2.0/robotwin2.0/meta/episodes.jsonl",
        help="Path to FastWAM processed episodes.jsonl.",
    )
    parser.add_argument(
        "--official-root",
        default="./data/robotwin2_official/dataset",
        help="Root directory containing official RoboTwin task folders.",
    )
    parser.add_argument(
        "--output",
        default="./configs/data/robotwin2_official_task_mapping.json",
        help="Output mapping file. Use `.json` for compact task-range format or `.jsonl` for detailed per-episode records.",
    )
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Optional task name filter for debugging, e.g. beat_block_hammer.",
    )
    parser.add_argument(
        "--limit-official-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of official episodes to scan, for debugging.",
    )
    parser.add_argument(
        "--print-top-k-ambiguous",
        type=int,
        default=5,
        help="When instruction containment is ambiguous, print the top-k candidates.",
    )
    return parser.parse_args()


def load_processed_episodes(path: Path):
    processed = {}
    instruction_to_episode_ids: dict[str, list[int]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            episode_id = int(rec["episode_index"])
            tasks = list(rec["tasks"])
            task_set = set(tasks)
            processed[episode_id] = {
                "episode_index": episode_id,
                "tasks": tasks,
                "task_set": task_set,
                "task_count": len(task_set),
                "length": int(rec["length"]),
            }
            for instruction in task_set:
                instruction_to_episode_ids[instruction].append(episode_id)

    return processed, instruction_to_episode_ids


def official_episode_index_from_path(path: Path) -> int:
    match = re.search(r"episode(\d+)\.json$", path.name)
    if match is None:
        raise ValueError(f"Failed to parse official episode index from path: {path}")
    return int(match.group(1))


def official_task_groups(root: Path) -> list[dict]:
    task_groups = []
    task_dirs = [path for path in root.iterdir() if path.is_dir()]
    task_dirs.sort(key=lambda path: path.name)
    for task_dir in task_dirs:
        task_name = task_dir.name
        episodes = []
        subset_dirs = [path for path in task_dir.iterdir() if path.is_dir()]
        subset_dirs.sort(key=lambda path: path.name)
        for subset_dir in subset_dirs:
            instructions_dir = subset_dir / "instructions"
            video_dir = subset_dir / "video"
            if not instructions_dir.is_dir():
                continue

            # FastWAM appears to preserve the raw lexicographic import order
            # (episode0, episode1, episode10, ...), not numeric order.
            files = [path for path in instructions_dir.glob("episode*.json") if path.is_file()]
            files.sort(key=lambda path: path.name)
            for source_rank_in_subset, instruction_file in enumerate(files):
                official_episode_index = official_episode_index_from_path(instruction_file)
                episodes.append(
                    {
                        "task_name": task_name,
                        "subset_name": subset_dir.name,
                        "official_episode_index": official_episode_index,
                        "instruction_file": instruction_file,
                        "video_file": video_dir / f"episode{official_episode_index}.mp4",
                        "source_file_name": instruction_file.name,
                        "source_rank_in_subset": source_rank_in_subset,
                    }
                )

        task_groups.append(
            {
                "task_name": task_name,
                "episodes": episodes,
            }
        )

    return task_groups


def load_official_seen_unseen(path: Path) -> tuple[list[str], list[str], set[str], set[str]]:
    with path.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    seen = list(rec.get("seen", []))
    unseen = list(rec.get("unseen", []))
    return seen, unseen, set(seen), set(unseen)


def find_matching_processed_candidates(
    *,
    seen_set: set[str],
    processed: dict[int, dict],
    instruction_to_episode_ids: dict[str, list[int]],
) -> list[dict]:
    overlap_counter: Counter[int] = Counter()
    for instruction in seen_set:
        for episode_id in instruction_to_episode_ids.get(instruction, []):
            overlap_counter[episode_id] += 1

    candidates = []
    for episode_id, overlap_count in overlap_counter.items():
        proc = processed[episode_id]
        proc_task_count = proc["task_count"]
        if overlap_count != proc_task_count:
            continue
        if not proc["task_set"].issubset(seen_set):
            continue
        candidates.append(
            {
                "processed_episode_index": episode_id,
                "processed_task_count": proc_task_count,
                "processed_length": proc["length"],
                "overlap_count": overlap_count,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["overlap_count"],
            item["processed_episode_index"],
        )
    )
    return candidates


def format_candidate_summary(candidate: dict) -> str:
    return (
        f"fastwam_episode={candidate['processed_episode_index']} "
        f"overlap={candidate['overlap_count']} "
        f"processed_task_count={candidate['processed_task_count']} "
        f"processed_length={candidate['processed_length']}"
    )


def _read_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">I", buf, offset)[0]


def _read_u64(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", buf, offset)[0]


def _iter_mp4_boxes(buf: bytes, start: int, end: int):
    offset = start
    while offset + 8 <= end:
        size = _read_u32(buf, offset)
        box_type = buf[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = _read_u64(buf, offset + 8)
            header_size = 16
        elif size == 0:
            size = end - offset

        if size < header_size or offset + size > end:
            break

        yield box_type, offset + header_size, offset + size
        offset += size


def _find_mp4_child(buf: bytes, start: int, end: int, target: bytes):
    for box_type, child_start, child_end in _iter_mp4_boxes(buf, start, end):
        if box_type == target:
            return child_start, child_end
    return None


def video_frame_count_minus_one(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Official video file not found: {path}")

    buf = path.read_bytes()
    moov = _find_mp4_child(buf, 0, len(buf), b"moov")
    if moov is None:
        raise ValueError(f"Failed to find moov box in: {path}")

    for box_type, trak_start, trak_end in _iter_mp4_boxes(buf, moov[0], moov[1]):
        if box_type != b"trak":
            continue
        mdia = _find_mp4_child(buf, trak_start, trak_end, b"mdia")
        if mdia is None:
            continue
        hdlr = _find_mp4_child(buf, mdia[0], mdia[1], b"hdlr")
        if hdlr is None:
            continue

        handler_type = buf[hdlr[0] + 8 : hdlr[0] + 12]
        if handler_type != b"vide":
            continue

        minf = _find_mp4_child(buf, mdia[0], mdia[1], b"minf")
        stbl = _find_mp4_child(buf, minf[0], minf[1], b"stbl") if minf is not None else None
        stsz = _find_mp4_child(buf, stbl[0], stbl[1], b"stsz") if stbl is not None else None
        if stsz is None:
            continue

        sample_count = _read_u32(buf, stsz[0] + 8)
        return sample_count - 1

    raise ValueError(f"Failed to find video sample table in: {path}")


def resolve_assignment(
    *,
    candidates: list[dict],
    expected_processed_episode_index: int,
    official_length: int,
    processed: dict[int, dict],
) -> tuple[dict | None, str, bool, bool]:
    candidate_ids = {item["processed_episode_index"] for item in candidates}
    expected_in_candidates = expected_processed_episode_index in candidate_ids
    length_match_to_expected = (
        expected_processed_episode_index in processed
        and processed[expected_processed_episode_index]["length"] == official_length
    )

    if expected_in_candidates and length_match_to_expected:
        chosen = next(
            item
            for item in candidates
            if item["processed_episode_index"] == expected_processed_episode_index
        )
        return chosen, "strict_order_length_match", True, True

    exact_length_candidates = [
        item for item in candidates if item["processed_length"] == official_length
    ]
    if len(exact_length_candidates) == 1:
        return exact_length_candidates[0], "unique_length_match", expected_in_candidates, length_match_to_expected

    if len(candidates) == 1:
        return candidates[0], "unique_instruction_match", expected_in_candidates, length_match_to_expected

    if exact_length_candidates:
        return exact_length_candidates[0], "first_length_match", expected_in_candidates, length_match_to_expected

    if candidates:
        return candidates[0], "first_instruction_match", expected_in_candidates, length_match_to_expected

    return None, "unmatched", expected_in_candidates, length_match_to_expected


def build_compact_task_mapping(mapping_records: list[dict]) -> dict:
    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for record in mapping_records:
        task_to_episodes[record["task_name"]].append(int(record["processed_episode_index"]))

    tasks = []
    for task_name in sorted(task_to_episodes.keys()):
        episode_indices = sorted(set(task_to_episodes[task_name]))
        if not episode_indices:
            continue
        for prev_ep, next_ep in zip(episode_indices, episode_indices[1:]):
            if next_ep != prev_ep + 1:
                raise ValueError(
                    f"Task {task_name} does not map to a contiguous episode range: "
                    f"{prev_ep}, {next_ep}"
                )
        tasks.append(
            {
                "task_name": task_name,
                "episode_start": episode_indices[0],
                "episode_end": episode_indices[-1],
                "episode_count": len(episode_indices),
            }
        )

    return {
        "format": "robotwin_task_ranges_v1",
        "total_tasks": len(tasks),
        "total_episodes": sum(task["episode_count"] for task in tasks),
        "tasks": tasks,
    }


def main() -> None:
    args = parse_args()
    processed_meta = Path(args.processed_meta).expanduser().resolve()
    official_root = Path(args.official_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not processed_meta.is_file():
        raise SystemExit(f"Processed episodes file not found: {processed_meta}")
    if not official_root.is_dir():
        raise SystemExit(f"Official dataset root not found: {official_root}")

    processed, instruction_to_episode_ids = load_processed_episodes(processed_meta)
    task_groups = official_task_groups(official_root)
    print(
        f"Loaded {len(processed)} FastWAM episodes from {processed_meta} "
        f"with {len(instruction_to_episode_ids)} unique instructions."
    )
    print(
        f"Loaded {len(task_groups)} official task groups from {official_root}."
    )

    mapping_records: list[dict] = []
    matched_processed_ids: set[int] = set()
    ambiguous_count = 0
    unmatched_count = 0
    strict_confidence_count = 0

    total_official_scanned = 0
    task_block_start = 0
    for task_group in task_groups:
        task_name = task_group["task_name"]
        task_episodes = task_group["episodes"]
        is_selected = args.task_filter is None or task_name == args.task_filter
        if is_selected:
            print(
                f"[task-block] task={task_name} block_start={task_block_start} "
                f"block_size={len(task_episodes)} block_end={task_block_start + len(task_episodes) - 1}"
            )

        for source_rank, official_episode in enumerate(task_episodes):
            if not is_selected:
                continue
            if args.limit_official_episodes is not None and total_official_scanned >= args.limit_official_episodes:
                break
            total_official_scanned += 1

            subset_name = official_episode["subset_name"]
            official_episode_index = official_episode["official_episode_index"]
            instruction_file = official_episode["instruction_file"]
            seen, unseen, seen_set, unseen_set = load_official_seen_unseen(instruction_file)
            candidates = find_matching_processed_candidates(
                seen_set=seen_set,
                processed=processed,
                instruction_to_episode_ids=instruction_to_episode_ids,
            )
            official_length = video_frame_count_minus_one(official_episode["video_file"])
            expected_processed_episode_index = task_block_start + source_rank

            chosen, resolution_stage, expected_in_candidates, length_match_to_expected = resolve_assignment(
                candidates=candidates,
                expected_processed_episode_index=expected_processed_episode_index,
                official_length=official_length,
                processed=processed,
            )

            prefix = (
                f"[match-check] task={task_name} subset={subset_name} "
                f"official_episode={official_episode_index} source_file={official_episode['source_file_name']} "
                f"source_rank={source_rank} expected_fastwam_episode={expected_processed_episode_index} "
                f"seen={len(seen_set)} unseen={len(unseen_set)} candidates={len(candidates)}"
            )

            if chosen is None:
                unmatched_count += 1
                print(
                    prefix
                    + (
                        " result=UNMATCHED "
                        f"official_length={official_length} "
                        f"expected_in_candidates={expected_in_candidates} "
                        f"length_match_to_expected={length_match_to_expected}"
                    )
                )
                continue

            strict_confidence = resolution_stage == "strict_order_length_match"
            if strict_confidence:
                strict_confidence_count += 1
            if len(candidates) > 1:
                ambiguous_count += 1

            assigned_processed = processed[chosen["processed_episode_index"]]
            from_seen = len(assigned_processed["task_set"] & seen_set)
            from_unseen = len(assigned_processed["task_set"] & unseen_set)

            print(
                prefix
                + (
                    " result=MATCH "
                    f"assigned_fastwam_episode={chosen['processed_episode_index']} "
                    f"resolution_stage={resolution_stage} "
                    f"{format_candidate_summary(chosen)} "
                    f"official_length={official_length} "
                    f"expected_in_candidates={expected_in_candidates} "
                    f"length_match_to_expected={length_match_to_expected} "
                    f"all_fastwam_in_seen={from_seen == chosen['processed_task_count']} "
                    f"from_seen={from_seen} "
                    f"from_unseen={from_unseen} "
                    f"strict_confidence={strict_confidence}"
                )
            )

            if len(candidates) > 1:
                preview = candidates[: args.print_top_k_ambiguous]
                for rank, candidate in enumerate(preview, start=1):
                    marker = " expected" if candidate["processed_episode_index"] == expected_processed_episode_index else ""
                    print(f"  [candidate {rank}] {format_candidate_summary(candidate)}{marker}")

            mapping_records.append(
                {
                    "processed_episode_index": chosen["processed_episode_index"],
                    "task_name": task_name,
                    "official_subset": subset_name,
                    "official_episode_index": official_episode_index,
                    "source_file_name": official_episode["source_file_name"],
                    "source_order_rank": source_rank,
                    "expected_processed_episode_index": expected_processed_episode_index,
                    "processed_task_count": chosen["processed_task_count"],
                    "processed_length": chosen["processed_length"],
                    "official_length": official_length,
                    "official_seen_count": len(seen_set),
                    "official_unseen_count": len(unseen_set),
                    "overlap_count": chosen["overlap_count"],
                    "from_seen_count": from_seen,
                    "from_unseen_count": from_unseen,
                    "all_fastwam_in_seen": from_seen == chosen["processed_task_count"],
                    "num_candidates": len(candidates),
                    "expected_in_candidates": expected_in_candidates,
                    "length_match_to_expected": length_match_to_expected,
                    "resolution_stage": resolution_stage,
                    "strict_confidence": strict_confidence,
                }
            )
            matched_processed_ids.add(chosen["processed_episode_index"])

        task_block_start += len(task_episodes)
        if args.limit_official_episodes is not None and total_official_scanned >= args.limit_official_episodes:
            break

    mapping_records.sort(key=lambda item: item["processed_episode_index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        compact_payload = build_compact_task_mapping(mapping_records)
        output_path.write_text(
            json.dumps(compact_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        with output_path.open("w", encoding="utf-8") as f:
            for record in mapping_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    duplicate_processed_ids = len(mapping_records) - len({r["processed_episode_index"] for r in mapping_records})
    missing_processed_ids = sorted(set(processed.keys()) - matched_processed_ids)

    print("\n[summary]")
    print(f"official_episodes_scanned={total_official_scanned}")
    print(f"mapping_records_written={len(mapping_records)}")
    print(f"strict_confidence_matches={strict_confidence_count}")
    print(f"instruction_ambiguous_episodes={ambiguous_count}")
    print(f"unmatched_official_episodes={unmatched_count}")
    print(f"duplicate_processed_episode_assignments={duplicate_processed_ids}")
    print(f"missing_processed_episode_count={len(missing_processed_ids)}")
    if missing_processed_ids:
        print(f"missing_processed_episode_preview={missing_processed_ids[:20]}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
