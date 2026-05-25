---
name: github-code-change-flow
description: End-to-end workflow for code changes in GitHub repositories: implement the change, validate it, commit with the required RelayAgent co-author trailer, push a branch, create or prepare a GitHub PR, run an automatic code review, iterate on review feedback, and merge to main when safe. Use this skill whenever the user asks to modify code, fix bugs, refactor, adjust tests, or change configuration and the current repository has a GitHub remote, even if the user only asks for the code change and does not explicitly mention PRs or merging.
---

# GitHub Code Change Flow

Use this skill to turn a GitHub-hosted code change into a complete delivery loop: code implementation, verification, Git commit, branch push, PR creation or preparation, automatic review, review-fix iteration, and safe merge to `main`.

## When to use

Use this skill when all of the following are true:

- The user requests a code implementation, bug fix, refactor, test change, configuration change, documentation change tied to code, or similar repository modification.
- The current working directory is a Git repository.
- The repository has a GitHub remote, typically detectable from `git remote -v` containing `github.com`.
- The user has not explicitly prohibited Git operations, PR creation, or merging.

If the user only says "改代码", "修一下", "实现这个功能", "fix", "refactor", or similar, and the repository is a GitHub project, default to this full workflow rather than stopping after local edits.

## Operating principles

- Keep the user-facing TODO LIST current for multi-step work. Show the full list at the start, and update it after completing each major stage.
- Make the smallest coherent change that satisfies the request.
- Keep the working tree clean except for files directly related to the task.
- Prefer explicit validation over assumptions: run relevant tests, syntax checks, linters, or focused smoke checks when available.
- Treat code review as a quality gate. Do not merge while review feedback is unresolved.
- Do not invent tool names. Use only tools available in the current environment.
- Do not hide failures. If a gate fails, stop and report the exact state and next safe options.

## Workflow

### 1. Confirm repository and intent

1. Inspect Git state and remotes:
   - `git status --short`
   - `git branch --show-current`
   - `git remote -v`
2. Confirm the repository has a GitHub remote.
3. Check whether the user has restricted GitHub, PR, or merge actions.
4. If unrelated local changes already exist, preserve them. Do not overwrite, stage, commit, or reformat unrelated files.

Pause and ask/report before proceeding if:

- The repository is not a Git repository.
- No GitHub remote exists.
- The user explicitly requested local-only changes.
- The working tree contains unrelated modifications that would be hard to isolate safely.

### 2. Plan the change

1. Restate the goal briefly.
2. Identify likely files and validation commands.
3. For non-trivial work, use code exploration before editing. If a `code-explorer` capability is available, use it for repository structure, call-flow, or impact analysis.
4. Avoid over-planning small obvious changes; proceed once there is enough context.

### 3. Implement

1. Edit only relevant files.
2. Keep changes focused and reversible.
3. Follow project conventions and existing style.
4. Update tests, fixtures, docs, or configuration when necessary for correctness.
5. Re-check `git diff` during implementation to ensure the patch is scoped.

### 4. Validate

Run the most relevant available checks, such as:

- Unit or integration tests for touched areas.
- Syntax checks or compilation.
- Linters or format checks if the project already uses them.
- Manual smoke checks for generated scripts or CLIs.

Record:

- Exact command executed.
- Pass/fail result.
- Any skipped checks and why they were skipped.

If validation fails, fix and rerun once the cause is understood. If tests still fail or the failure appears unrelated but risky, stop and report.

### 5. Commit with required trailer

Before committing:

1. Review `git status --short` and `git diff`.
2. Stage only task-related files.
3. Use a clear Conventional Commits style message when appropriate.
4. Include this exact trailer at the end of the commit message:

```text
Co-Authored-By: RelayAgent <noreply@relayagent.local>
```

Example:

```text
fix: handle missing docker image argument

Add validation for required image input and improve the generated error message.

Co-Authored-By: RelayAgent <noreply@relayagent.local>
```

After committing, verify the latest commit message includes the trailer:

```bash
git log -1 --format=full
```

### 6. Push a branch

1. Create a task branch unless already on an appropriate non-main branch.
2. Use a descriptive branch name, for example:
   - `fix/<short-topic>`
   - `feat/<short-topic>`
   - `chore/<short-topic>`
3. Push the branch to the GitHub remote:

```bash
git push -u origin <branch-name>
```

If push is denied because of authentication or permission issues, stop and report the branch name, commit hash, and exact push failure.

### 7. Create or prepare the PR

Prefer the GitHub CLI when available:

```bash
gh pr create --base main --head <branch-name> --title "<title>" --body "<body>"
```

PR title/body should include:

- Goal.
- Key changes.
- Tests or checks run.
- Automatic review conclusion.
- Risks and rollback plan.

If `gh` is unavailable but `git push` succeeded:

1. Provide the pushed branch name.
2. Provide the likely GitHub compare URL when it can be derived from the remote, for example:
   `https://github.com/<owner>/<repo>/compare/main...<branch-name>`
3. Provide a ready-to-copy PR title and body.

### 8. Automatic code review

Run an automatic review before merge. Prefer an available repository-analysis or code-review capability. If the environment provides `code-explorer`, use it to inspect the diff and review for correctness, regressions, missing tests, security issues, and maintainability.

Review focus:

- Does the implementation satisfy the user request?
- Are edge cases handled?
- Are tests or validation sufficient for the change size?
- Is the diff limited to relevant files?
- Are credentials, secrets, generated artifacts, or unrelated changes accidentally included?
- Is the rollback path straightforward?

If review finds issues:

1. Fix the issues in the same branch.
2. Rerun relevant validation.
3. Commit follow-up fixes with the same `Co-Authored-By` trailer.
4. Push the branch again.
5. Repeat review until it passes or a blocking issue remains.

### 9. Merge to main when safe

Merge only when all safe-merge conditions are met:

- Validation passed or skipped with a clearly acceptable reason.
- Automatic review passed.
- No unresolved merge conflicts.
- No high-risk or destructive changes require explicit user approval.
- The user has not prohibited merge.
- The repository permissions allow merge or push to `main`.

Preferred merge path when `gh` is available and a PR exists:

```bash
gh pr merge <pr-number-or-url> --merge --delete-branch
```

Use the repository's established strategy if visible, such as squash or rebase, instead of forcing a different one.

Fallback when `gh` is unavailable, the user expects automatic merge, and local `main` can be pushed safely:

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git merge --no-ff <branch-name>
git push origin main
```

After merge:

1. Confirm `main` is updated.
2. Confirm remote push succeeded.
3. Report whether the feature branch was deleted or remains.

## Safety boundaries

Pause and ask the user or report instead of continuing when encountering:

- Force push requirements, especially to shared branches.
- Deleting branches, files, tags, releases, or remote resources when not clearly part of the task.
- Production deployment or release operations.
- Permission or authentication failures.
- Merge conflicts that cannot be resolved with high confidence.
- Test failures, syntax failures, or lint failures that are not safely explainable.
- Automatic review failure.
- Unexpected unrelated working-tree changes.
- Changes involving secrets, credentials, access control, payments, destructive migrations, or security-sensitive behavior.
- Any user instruction that limits the workflow, such as "不要提交", "不要 PR", "先别 merge", or "local only".

Never use force push, destructive cleanup, or broad reset commands unless the user explicitly approves after seeing the risk.

## Required final output format

At completion or safe stop, report these sections:

```markdown
## ✅ 完成状态
- Status: completed | paused | failed
- Reason: <brief reason>

## [📋 TODO LIST]
🚀 进度: <done> / <total>
✅ [1] ...
✅ [2] ...

## 变更文件
- `<path>`: <summary>

## 验证结果
- `<command>`: passed | failed | skipped — <notes>

## Commit
- Branch: `<branch>`
- Commit: `<hash>`
- Message: `<subject>`
- Co-Authored-By trailer: present | missing

## PR / Branch
- PR: <url or not created>
- Branch: <remote branch or local branch>
- PR body includes: goal, changes, tests, review conclusion, risks/rollback

## 自动检视结论
- Result: passed | failed | skipped
- Findings: <summary>

## Merge 状态
- Merged to main: yes | no
- Method: gh PR merge | local merge and push | not merged
- Main pushed: yes | no
- Remaining action: <if any>
```

If the workflow stops early, still include the same sections and clearly identify what is complete, what is blocked, and the safest next step.
