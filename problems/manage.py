#!/usr/bin/env python3
"""
SETU Problem & Plan Lifecycle Manager
====================================
Automates tracking, movement, and verification of architectural problems,
accuracy issues, and plans across their 4 distinct lifecycle stages:
  - plan/        : Initial proposals, plans, and architectural designs
  - not_solved/  : Accepted open problems pending implementation
  - in_progress/ : Active problems currently being implemented
  - solved/      : Verified, tested, and completed problems

Usage:
  python problems/manage.py list
  python problems/manage.py move <ID-or-pattern> <plan|not_solved|in_progress|solved>
  python problems/manage.py new <ID> <slug> "<Title>"
  python problems/manage.py test
"""

import sys
import os
import re
import shutil
import argparse
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("plan", "not_solved", "in_progress", "solved")
ROOT_DIR = Path(__file__).resolve().parent

def get_stage_dir(stage: str, base_dir: Path = ROOT_DIR) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {', '.join(STAGES)}")
    stage_dir = base_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir

def find_problem_file(pattern: str, base_dir: Path = ROOT_DIR) -> tuple[Path, str]:
    """Find a problem markdown file across all stages by ID or substring."""
    pattern_lower = pattern.lower().strip()
    matches = []
    for stage in STAGES:
        stage_dir = base_dir / stage
        if not stage_dir.exists():
            continue
        for file in stage_dir.glob("*.md"):
            if pattern_lower in file.name.lower():
                matches.append((file, stage))
            else:
                # Check inside frontmatter for id: PROB-XX
                try:
                    content = file.read_text(encoding="utf-8")
                    if re.search(rf"\bid:\s*['\"]?{re.escape(pattern_lower)}['\"]?", content, re.IGNORECASE):
                        matches.append((file, stage))
                except Exception:
                    pass

    if not matches:
        raise FileNotFoundError(f"No problem file found matching '{pattern}' in any stage folder.")
    if len(matches) > 1:
        # Check if one is an exact ID match
        exact = [m for m in matches if pattern_lower in m[0].stem.lower().split("-")]
        if len(exact) == 1:
            return exact[0]
        match_names = [f"{m[1]}/{m[0].name}" for m in matches]
        raise ValueError(f"Ambiguous pattern '{pattern}' matched multiple files: {match_names}")
    return matches[0]

def update_frontmatter(content: str, new_status: str) -> str:
    """Update or inject Status and Last-Updated in Markdown frontmatter."""
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Replace Status line if present
    if re.search(r"^status:\s*.*$", content, flags=re.MULTILINE | re.IGNORECASE):
        content = re.sub(
            r"^status:\s*.*$",
            f"status: {new_status}",
            content,
            flags=re.MULTILINE | re.IGNORECASE
        )
    # Replace updated date if present
    if re.search(r"^updated:\s*.*$", content, flags=re.MULTILINE | re.IGNORECASE):
        content = re.sub(
            r"^updated:\s*.*$",
            f"updated: {now_iso}",
            content,
            flags=re.MULTILINE | re.IGNORECASE
        )
    return content

def move_problem(pattern: str, target_stage: str, base_dir: Path = ROOT_DIR) -> Path:
    """Move a problem card from its current stage to the target stage."""
    target_dir = get_stage_dir(target_stage, base_dir)
    src_file, current_stage = find_problem_file(pattern, base_dir)
    
    if current_stage == target_stage:
        print(f"[*] '{src_file.name}' is already in '{target_stage}'. Updating metadata...")
        content = src_file.read_text(encoding="utf-8")
        updated = update_frontmatter(content, target_stage)
        src_file.write_text(updated, encoding="utf-8")
        return src_file

    dest_file = target_dir / src_file.name
    content = src_file.read_text(encoding="utf-8")
    updated = update_frontmatter(content, target_stage)
    
    dest_file.write_text(updated, encoding="utf-8")
    src_file.unlink()
    
    print(f"[+] Moved '{src_file.name}' : {current_stage} -> {target_stage}")
    return dest_file

def new_problem(prob_id: str, slug: str, title: str, stage: str = "plan", base_dir: Path = ROOT_DIR) -> Path:
    """Create a new problem card in plan/ (or designated stage)."""
    target_dir = get_stage_dir(stage, base_dir)
    prob_id = prob_id.upper()
    filename = f"{prob_id}-{slug}.md"
    file_path = target_dir / filename
    
    if file_path.exists():
        raise FileExistsError(f"Problem file already exists: {file_path}")
        
    now_date = datetime.now().strftime("%Y-%m-%d")
    template = f"""---
id: {prob_id}
title: "{title}"
stage: {stage}
status: {stage}
created: {now_date}
updated: {now_date}
severity: High
category: Accuracy & Pipeline
---

# {prob_id}: {title}

## 1. Problem Description
<!-- Clear summary of the symptom and observed failure mode -->

## 2. Root Cause Analysis
<!-- Technical deep-dive into the exact files, lines, and math causing this -->

## 3. Implementation Plan & Proposed Fix
<!-- Step-by-step code and architectural changes needed to solve this -->

## 4. Acceptance Criteria & Verification
<!-- Exact tests or quantitative benchmarks required before moving to 'solved' -->
- [ ] Targeted test implemented and passing
- [ ] PyTest test suite remains 100% green
- [ ] No regression on existing sign classifications
"""
    file_path.write_text(template, encoding="utf-8")
    print(f"[+] Created new problem card: {stage}/{filename}")
    return file_path

def list_problems(base_dir: Path = ROOT_DIR):
    """Print an ASCII dashboard of problems across all 4 stages."""
    print("=" * 78)
    print("                      SETU PROBLEMS & PLANS DASHBOARD                     ")
    print("=" * 78)
    
    total = 0
    stage_counts = {}
    for stage in STAGES:
        stage_dir = base_dir / stage
        files = sorted(list(stage_dir.glob("*.md"))) if stage_dir.exists() else []
        stage_counts[stage] = len(files)
        total += len(files)
        
        stage_badge = {
            "plan": "[ PLAN / PROPOSAL ]",
            "not_solved": "[ NOT SOLVED ]",
            "in_progress": "[ IN PROGRESS ]",
            "solved": "[ SOLVED & VERIFIED ]"
        }[stage]
        
        print(f"\n{stage_badge} ({len(files)} items)")
        print("-" * 78)
        if not files:
            print("  (None)")
        for f in files:
            title = f.stem
            # Attempt to read title from YAML frontmatter
            try:
                txt = f.read_text(encoding="utf-8")
                m = re.search(r"^title:\s*['\"]?(.*?)['\"]?$", txt, flags=re.MULTILINE)
                if m:
                    title = m.group(1)
            except Exception:
                pass
            print(f"  * {f.name:<40} | {title[:32]}")
            
    print("\n" + "=" * 78)
    print(f"Summary: {total} total | Plan: {stage_counts['plan']} | Open: {stage_counts['not_solved']} | Active: {stage_counts['in_progress']} | Solved: {stage_counts['solved']}")
    print("=" * 78)

def run_self_test():
    """Ponytail self-check: verify manage.py logic using a temporary sandbox."""
    import tempfile
    print("[*] Running manage.py self-test...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # 1. Test creation in plan
        card = new_problem("PROB-99", "test-issue", "Test Issue for Automation", stage="plan", base_dir=tmp_path)
        assert card.exists(), "Created card does not exist"
        assert card.parent.name == "plan", "Card not in plan/"
        
        # 2. Test search
        found, stage = find_problem_file("PROB-99", base_dir=tmp_path)
        assert stage == "plan", "Found wrong stage"
        assert found == card, "Found wrong file"
        
        # 3. Test move to not_solved
        card2 = move_problem("PROB-99", "not_solved", base_dir=tmp_path)
        assert card2.exists(), "Moved card does not exist"
        assert card2.parent.name == "not_solved", "Card not in not_solved/"
        assert not card.exists(), "Old card still exists in plan/"
        assert "status: not_solved" in card2.read_text(encoding="utf-8")
        
        # 4. Test move to in_progress
        card3 = move_problem("test-issue", "in_progress", base_dir=tmp_path)
        assert card3.parent.name == "in_progress"
        assert "status: in_progress" in card3.read_text(encoding="utf-8")
        
        # 5. Test move to solved
        card4 = move_problem("99", "solved", base_dir=tmp_path)
        assert card4.parent.name == "solved"
        assert "status: solved" in card4.read_text(encoding="utf-8")
        
    print("[+] All manage.py self-tests PASSED successfully!")

def main():
    parser = argparse.ArgumentParser(description="SETU Problem Lifecycle Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # list
    subparsers.add_parser("list", help="List all problems and plans across stages")
    subparsers.add_parser("status", help="Alias for list")
    
    # move
    p_move = subparsers.add_parser("move", help="Move a problem to a different stage")
    p_move.add_argument("id", help="Problem ID (e.g. PROB-01) or file substring")
    p_move.add_argument("stage", choices=STAGES, help="Target stage")
    
    # new
    p_new = subparsers.add_parser("new", help="Create a new problem card")
    p_new.add_argument("id", help="Problem ID, e.g. PROB-11")
    p_new.add_argument("slug", help="Slug for filename, e.g. camera-latency")
    p_new.add_argument("title", help="Human readable title")
    p_new.add_argument("--stage", choices=STAGES, default="plan", help="Initial stage (default: plan)")
    
    # test
    subparsers.add_parser("test", help="Run self-checks")
    
    args = parser.parse_args()
    
    if args.command in ("list", "status") or not args.command:
        list_problems()
    elif args.command == "move":
        move_problem(args.id, args.stage)
    elif args.command == "new":
        new_problem(args.id, args.slug, args.title, stage=args.stage)
    elif args.command == "test":
        run_self_test()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

