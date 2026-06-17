# Git & Source Code Management — Interview Notes

> **Covers:** Core concepts · Branching · Merging · Workflows · Collaboration · Advanced Topics · Best Practices

---

## 1. What is Git? Why is Version Control Important?

**Git** is a distributed version control system (DVCS) created by Linus Torvalds in 2005. It tracks changes in source code over time, enabling multiple developers to collaborate without overwriting each other's work.

**Why it matters:**
- Full history of every change made to the codebase
- Ability to revert to any previous state
- Parallel development via branches
- Collaboration without a single point of failure (distributed model)

**Centralized vs Distributed VCS:**

| Feature | Centralized (SVN) | Distributed (Git) |
|---|---|---|
| Repository location | Single server | Every developer has a full copy |
| Offline work | Limited | Fully supported |
| Speed | Slower (network calls) | Faster (local operations) |
| Branching | Heavy, expensive | Lightweight, instant |

---

## 2. Core Git Concepts

### 2.1 The Three Areas

```
Working Directory  →  Staging Area (Index)  →  Repository (.git)
     (edit files)        (git add)                (git commit)
```

- **Working Directory** — where you edit files on disk
- **Staging Area (Index)** — a preparation zone; files you've marked for the next commit
- **Repository** — the `.git` folder storing all history and objects

**Example flow:**
```bash
# Edit a file
echo "Hello World" > app.js

# Stage it
git add app.js

# Commit it
git commit -m "Add initial app.js"
```

### 2.2 The Four Object Types

| Object | Description |
|---|---|
| **Blob** | Content of a file (no name, no path) |
| **Tree** | Directory listing (maps names → blobs/trees) |
| **Commit** | Snapshot + metadata (author, message, parent) |
| **Tag** | Named pointer to a specific commit |

### 2.3 HEAD

`HEAD` is a pointer to the currently checked-out commit or branch. It tells Git "where you are right now."

```bash
cat .git/HEAD
# → ref: refs/heads/main   (on a branch)
# → a3f9c12...             (detached HEAD)
```

---

## 3. Essential Git Commands

### 3.1 Setup & Init

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

git init                    # Start a new local repo
git clone <url>             # Clone a remote repo
git clone <url> my-folder   # Clone into specific folder
```

### 3.2 Staging & Committing

```bash
git status                  # Show working tree status
git add file.txt            # Stage a specific file
git add .                   # Stage all changes
git add -p                  # Interactively stage chunks (very useful!)

git commit -m "message"     # Commit with inline message
git commit --amend          # Edit the last commit (message or content)
```

### 3.3 Viewing History

```bash
git log                     # Full commit history
git log --oneline           # Compact one-line per commit
git log --oneline --graph   # Visual branch graph
git log -p                  # Show diffs with each commit
git log --author="Alice"    # Filter by author
git log --since="2 weeks ago"

git diff                    # Working dir vs staging
git diff --staged           # Staging vs last commit
git diff main..feature      # Compare two branches
git show a3f9c12            # Show a specific commit
```

### 3.4 Undoing Changes

```bash
git restore file.txt        # Discard working dir changes (untracked)
git restore --staged file.txt  # Unstage a file

git revert <commit>         # Create a new commit that undoes a commit (safe, preserves history)
git reset --soft HEAD~1     # Undo last commit, keep changes staged
git reset --mixed HEAD~1    # Undo last commit, keep changes unstaged (default)
git reset --hard HEAD~1     # Undo last commit AND discard all changes (DESTRUCTIVE)
```

> **Interview tip:** `revert` is safe for shared branches; `reset` rewrites history — never use `reset --hard` on pushed commits.

---

## 4. Branching

Branches are lightweight, movable pointers to a commit. Creating a branch costs almost nothing.

### 4.1 Branch Commands

```bash
git branch                    # List all local branches
git branch -a                 # List all branches (including remote)
git branch feature-login      # Create a new branch
git checkout feature-login    # Switch to branch
git switch feature-login      # Modern way to switch (Git 2.23+)
git switch -c feature-login   # Create AND switch

git branch -d feature-login   # Delete merged branch (safe)
git branch -D feature-login   # Force delete (even if unmerged)

git branch -m old-name new-name  # Rename a branch
```

### 4.2 Branching Strategy (Interview Key Topic)

**Feature Branch Workflow:**
```
main
  └── feature/user-auth
  └── feature/payment-gateway
  └── bugfix/login-crash
```
Each feature lives in its own branch → merged to main when done.

**Git Flow (common in enterprises):**
```
main (production)
  └── develop (integration)
        └── feature/xxx    → merge into develop
        └── release/1.2    → merge into main + develop
        └── hotfix/critical → merge into main + develop
```

**GitHub Flow (simpler, CI/CD friendly):**
```
main (always deployable)
  └── any-feature-branch  → PR → code review → merge to main → deploy
```

---

## 5. Merging

### 5.1 Fast-Forward Merge

Happens when the target branch has no new commits since the feature branch diverged. Git simply moves the pointer forward.

```bash
git checkout main
git merge feature-login
# Fast-forward: no merge commit created
```

```
Before:  main → A → B
                      └── feature → C → D

After:   main → A → B → C → D
```

### 5.2 Three-Way Merge (Merge Commit)

When both branches have diverged, Git creates a new "merge commit" with two parents.

```bash
git merge feature-login
# Creates commit M with parents B and D
```

```
Before:   A → B → C  (main)
               └── D → E  (feature)

After:    A → B → C → M  (main)
               └── D → E ↗
```

### 5.3 Rebase

Moves or replays commits from one branch onto another, creating a **linear history**.

```bash
git checkout feature-login
git rebase main
# Replays feature commits on top of latest main
```

```
Before:   A → B → C  (main)
               └── D → E  (feature)

After:    A → B → C → D' → E'  (feature, rebased)
```

**Rebase vs Merge:**

| | Merge | Rebase |
|---|---|---|
| History | Preserves full history with merge commits | Creates clean, linear history |
| Safety | Safe on shared branches | **Never rebase shared/public branches** |
| Use case | Integrating feature into main | Keeping a feature branch up to date |

### 5.4 Merge Conflicts

Occur when two branches change the same lines in a file.

```
<<<<<<< HEAD
  return "Hello from main";
=======
  return "Hello from feature";
>>>>>>> feature-login
```

**Resolution steps:**
```bash
# 1. Open the file and edit to desired state
# 2. Remove conflict markers
# 3. Stage the resolved file
git add app.js
# 4. Complete the merge
git commit
```

```bash
git mergetool      # Use a visual merge tool
git merge --abort  # Cancel an in-progress merge
```

---

## 6. Remote Repositories

### 6.1 Working with Remotes

```bash
git remote -v                          # List remotes
git remote add origin <url>            # Add a remote
git remote remove origin               # Remove a remote
git remote rename origin upstream      # Rename

git fetch origin           # Download changes, don't merge
git pull origin main       # fetch + merge (or rebase with --rebase flag)
git push origin main       # Push local commits to remote
git push -u origin main    # Set upstream tracking + push
git push --force-with-lease  # Safer force push (checks for remote changes first)
```

### 6.2 Fetch vs Pull

```
git fetch  → downloads commits but doesn't touch your working directory
git pull   → git fetch + git merge (or rebase)
```

**Best practice:** Use `git fetch` then review with `git log origin/main` before merging.

### 6.3 Tracking Branches

```bash
git branch -u origin/main main   # Set local main to track origin/main
git branch -vv                   # Show tracking info for all branches
```

---

## 7. Stashing

Temporarily shelve changes so you can switch context without committing.

```bash
git stash                   # Stash working dir + staged changes
git stash push -m "WIP login fix"  # Stash with a message
git stash list              # View all stashes
git stash pop               # Apply latest stash + remove it
git stash apply stash@{2}   # Apply a specific stash (keep it in list)
git stash drop stash@{0}    # Delete a stash
git stash clear             # Delete all stashes

git stash branch feature-from-stash  # Create a branch from stash
```

**Example scenario:**
```bash
# You're mid-feature and need to fix an urgent bug
git stash push -m "WIP: user dashboard"
git checkout bugfix/critical
# ... fix the bug, commit, push ...
git checkout feature/dashboard
git stash pop
# Continue where you left off
```

---

## 8. Tags

Tags mark specific points in history — typically used for releases.

```bash
git tag v1.0.0              # Lightweight tag (just a pointer)
git tag -a v1.0.0 -m "Release 1.0.0"  # Annotated tag (has metadata)
git tag                     # List all tags
git tag -l "v1.*"           # Filter tags
git push origin v1.0.0      # Push a specific tag
git push origin --tags      # Push all tags
git tag -d v1.0.0           # Delete local tag
git push origin --delete v1.0.0  # Delete remote tag
```

**Lightweight vs Annotated:**

| | Lightweight | Annotated |
|---|---|---|
| Stored as | Simple pointer | Full Git object |
| Has metadata | No | Yes (tagger, date, message) |
| Use case | Temporary marks | Official releases |

---

## 9. Advanced Git Operations

### 9.1 Cherry-Pick

Apply a specific commit from one branch to another.

```bash
git cherry-pick a3f9c12          # Apply a single commit
git cherry-pick a3f9c12 b7d2e44  # Apply multiple commits
git cherry-pick A..B             # Apply a range of commits
git cherry-pick --no-commit a3f9c12  # Stage without committing
```

**Use case:** A critical bug was fixed in `develop` and you need it in `hotfix` without merging the entire branch.

### 9.2 Git Bisect

Binary search through commit history to find which commit introduced a bug.

```bash
git bisect start
git bisect bad               # Current commit is broken
git bisect good v1.0.0       # Last known good commit

# Git checks out a middle commit
# Test it, then:
git bisect good   # or
git bisect bad

# Git narrows down until it finds the culprit
git bisect reset  # End the bisect session
```

### 9.3 Interactive Rebase

Rewrite, reorder, squash, or edit commits before pushing.

```bash
git rebase -i HEAD~3   # Interactively edit last 3 commits
```

In the editor:
```
pick a1b2c3 Add login form
pick d4e5f6 Fix typo in login form
pick g7h8i9 Add unit tests for login

# Commands:
# pick   = use commit as-is
# reword = change commit message
# squash = merge into previous commit (keep both messages)
# fixup  = merge into previous commit (discard this message)
# drop   = remove the commit entirely
```

**Squashing example:** Merge the first two commits into one clean commit before opening a PR.

### 9.4 Reflog

Git's safety net — records every change to HEAD, even after resets or rebases.

```bash
git reflog                  # Show all HEAD movements
git checkout HEAD@{3}       # Go back to a previous HEAD state
git reset --hard HEAD@{2}   # Recover from an accidental hard reset
```

> **Interview tip:** "Nothing is truly lost in Git within 90 days (default GC window). Use `git reflog` to recover."

### 9.5 Worktrees

Check out multiple branches simultaneously in different directories.

```bash
git worktree add ../hotfix-branch hotfix/critical-bug
# Now you have two directories: main project + hotfix folder
git worktree list
git worktree remove ../hotfix-branch
```

---

## 10. .gitignore

Tells Git which files to never track.

```gitignore
# Dependencies
node_modules/
vendor/

# Build output
dist/
build/
*.o
*.class

# Environment files
.env
.env.local
*.env

# OS files
.DS_Store
Thumbs.db

# IDE files
.idea/
.vscode/
*.swp

# Logs
*.log
logs/
```

```bash
git check-ignore -v filename    # Debug: why is this file being ignored?
git rm --cached file.txt        # Stop tracking a file (already committed)
```

---

## 11. Pull Requests & Code Review

A **Pull Request (PR)** / **Merge Request (MR)** is a formal proposal to merge a branch, enabling code review before integration.

**Ideal PR workflow:**
1. Create a feature branch from `main`
2. Make focused, atomic commits
3. Push the branch to remote
4. Open a PR with clear title and description
5. Automated CI checks run (tests, linting)
6. Reviewers leave comments
7. Author addresses feedback with new commits
8. Approvals received → Merge
9. Branch deleted

**Good PR practices:**
- Keep PRs small and focused (< 400 lines ideally)
- Write a clear description: what changed and why
- Reference related issues (`Closes #123`)
- Don't mix unrelated changes

---

## 12. Git Hooks

Scripts that run automatically on Git events — great for enforcing standards.

```bash
# Location: .git/hooks/
pre-commit       # Run before a commit is created (linting, tests)
commit-msg       # Validate commit message format
pre-push         # Run before pushing (run test suite)
post-merge       # Run after a merge (e.g., npm install)
```

**Example pre-commit hook (enforce lint):**
```bash
#!/bin/sh
npm run lint
if [ $? -ne 0 ]; then
  echo "Linting failed. Commit aborted."
  exit 1
fi
```

```bash
chmod +x .git/hooks/pre-commit   # Make it executable
```

Use **Husky** (Node.js) to share hooks across the team via `package.json`.

---

## 13. Commit Message Best Practices

Follow the **Conventional Commits** or **Git commit guidelines**:

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples:**
```
feat(auth): add JWT-based authentication

fix(cart): prevent duplicate item additions

docs(readme): update installation instructions

refactor(api): extract validation into middleware

BREAKING CHANGE: auth tokens are now short-lived (1hr)
```

**Rules:**
- Subject line: 50 characters max, imperative mood ("Add" not "Added")
- Body: wrap at 72 characters
- Explain *why*, not *what* (the diff shows what)

---

## 14. Monorepo vs Polyrepo

| | Monorepo | Polyrepo |
|---|---|---|
| Definition | All projects in one repo | Each project in its own repo |
| Examples | Google, Meta, Nx, Turborepo | Most open-source projects |
| Pros | Atomic cross-project changes, shared tooling | Clear ownership, smaller scope |
| Cons | Tooling complexity, slow CI | Cross-team changes need multiple PRs |

---

## 15. Common Interview Questions & Answers

**Q: What is the difference between `git pull` and `git fetch`?**
`git fetch` downloads remote changes but doesn't modify your working branch. `git pull` is a shortcut for `git fetch` followed by `git merge` (or `git rebase`). Using `fetch` first lets you inspect what changed before integrating.

---

**Q: How do you resolve a merge conflict?**
Open the conflicting file, locate conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), edit to the desired state, remove markers, `git add` the file, then `git commit` (or `git merge --continue`).

---

**Q: What is a detached HEAD state?**
When `HEAD` points directly to a commit SHA instead of a branch name. Any commits made here won't belong to a branch and can be lost. Fix with `git switch -c new-branch` to save work, or `git switch main` to abandon.

---

**Q: Difference between `git reset` and `git revert`?**
`git reset` moves the branch pointer backward — it rewrites history (unsafe for shared branches). `git revert` creates a new commit that undoes a previous commit — it's safe for shared/public branches because history is preserved.

---

**Q: When would you use `git cherry-pick`?**
When you need a specific commit (e.g., a bug fix) from one branch applied to another without merging the entire branch.

---

**Q: What is `git stash` and when is it useful?**
`git stash` temporarily shelves changes so you can switch branches with a clean working directory, then restore them later with `git stash pop`.

---

**Q: How do you undo the last commit without losing changes?**
```bash
git reset --soft HEAD~1   # Keeps changes staged
git reset --mixed HEAD~1  # Keeps changes unstaged (default)
```

---

**Q: What is `git rebase -i` and what are its risks?**
Interactive rebase lets you edit, squash, reorder, or drop commits. Risk: it rewrites commit SHAs — never use on commits already pushed to a shared branch, as it will require force-pushing and break teammates' histories.

---

## 16. Quick Reference Cheat Sheet

```bash
# === SETUP ===
git init | git clone <url>

# === DAILY WORKFLOW ===
git status | git add . | git commit -m "msg"
git pull origin main | git push origin feature

# === BRANCHING ===
git switch -c feature/xyz    # Create + switch
git switch main              # Switch back
git merge feature/xyz        # Merge
git branch -d feature/xyz    # Clean up

# === UNDOING ===
git restore file.txt         # Discard working dir changes
git reset --soft HEAD~1      # Undo commit, keep staged
git revert <sha>             # Safe undo (new commit)

# === INSPECTION ===
git log --oneline --graph
git diff main..feature
git blame file.txt           # Who wrote which line

# === REMOTE ===
git fetch origin
git push -u origin feature
git push --force-with-lease  # Safer force push

# === ADVANCED ===
git stash push -m "WIP"
git stash pop
git cherry-pick <sha>
git bisect start/bad/good
git reflog
git rebase -i HEAD~3
```

---

*Good luck with your interview! Remember: demonstrate understanding of **why** each command exists, not just the syntax.*
