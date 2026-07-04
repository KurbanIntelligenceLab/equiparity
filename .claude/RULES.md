# RULES.md

## Decisions

Ask the user before any change that affects behavior, architecture, public API, experiment results, paper claims, or user-facing output. Do not decide these alone. When a request requires user judgment, ask for clarification before acting.

Auto-fix mechanical issues without waiting for approval: typing, linting, formatting, import ordering, and similar changes that are clearly correct and behavior-neutral. If a fix that appeared mechanical turns out to alter behavior, results, or tests, revert it and flag it to the user instead.

When a rule about mechanical fixes and the ask-first rule both apply to the same change, ask first. The ask-first rules win on conflict.

## Verification

Passing tests is not proof of correctness. Verify the result logically as well as mechanically. When a task or todo item is done, confirm the work is correct and the output is what was expected, and report to the user when it is not.

Notify the user of any mismatch, inconsistency, or unexpected behavior. Never silently ignore them.

Implement proper fixes. Do not use band-aids, temporary workarounds, or superficial fixes unless the user explicitly approves them.

## Reproducibility

Treat reproducibility as required, not optional.

For research repositories, confirm that dependencies, setup commands, training commands, evaluation commands, and expected outputs are documented. Document required datasets, model checkpoints, hardware, software versions, seeds, environment variables, and expected runtime when they affect results.

Verify that documented commands match the current code and are not stale. If the paper claims a result, the code should make clear how that result was produced or verified.

Do not claim that code, data, models, checkpoints, or results are available unless they exist at the stated location. If a result cannot be reproduced, report that clearly. Do not hide it.

When results, tables, figures, benchmarks, or claims change, check whether the README, paper text, experiment scripts, result tables, or reproduction instructions also need updating.

## Commits

Do not commit using your name. Do not add co-authoring or AI attribution to commits.

## Maintenance

After a task, check whether RULES.md or CLAUDE.md needs updating. Propose the change and ask before editing RULES.md; apply CLAUDE.md updates that are clearly correct.

## Comments and formatting

Keep comments short, direct, and boring across code, documentation, and `.tex` files. A comment explains what a line of code does or why; it is not a banner.

Do not add decorative separators, symbol runs, ASCII-art dividers, or padding characters, such as:

```
%%========================================================
%% SOME TEXT...
%%========================================================
```

Do not add long block comments, fancy formatting, or elaborate headers. No structural or navigational comments that just announce sections or restate the obvious.

For `.tex` files, do not add section guideline, direction, or instructional comments (for example `% --- Introduction: state the problem here ---`). Write the content directly without scaffolding comments around it.