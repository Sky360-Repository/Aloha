# Git quick guide

## Setup
* `git config --global user.name "Your Name"` – Set your name
* `git config --global user.email "you@example.com"` – Set your email
* `git clone <repo-url>` – Clone a remote repository

## Daily Workflow
* `git pull` – Get the latest changes from the remote
* `git status` – Check the state of your working directory
* `git add <file>` – Stage changes
* `git add .` – Add all changes
* `git commit -m "Meaningful message"` – Save a snapshot of changes
* `git push` – Upload your changes to the remote
* `git fetch` – Safely update remote tracking branches (no merge!)
* `git checkout -f -B main origin/main` – Hard-reset your local branch to match `origin/main`

## Branching
* `git branch` – See current branches
* `git checkout -b <branch-name>` – Create and switch to a new branch
* `git checkout <branch-name>` – Switch to an existing branch
* `git merge <branch>` – Merge another branch into your current one

# Undo & Fix
* `git restore <file>` – Discard local changes (not staged)
* `git reset HEAD <file>` – Unstage a file
* `git commit --amend` – Edit the last commit
* `git log` – View commit history
* `git reset --soft HEAD~1` – Undo last commit, keep changes staged
* `git reset --hard HEAD~1` – Danger! Reset to previous commit, erase changes
* `git citool` – Visual staging and commit interface
  * Easy to see staged vs. unstaged changes
  * Encourages writing clear commit messages with headers
  * Great for learning what Git is actually tracking

## Merging vs. Rebasing
* `git merge <branch>` – Merge changes with a new merge commit
  *(Keeps full history, but can get messy)*
* `git rebase <branch>` – Replay commits on top of another branch
  *(Cleaner, linear history)*
* `git rebase -i HEAD~n` – Interactive rebase for editing, reordering, squashing commits
  *(More visual, intuitive for learning how commits are structured)*

## **Advanced Recovery**
* `git reflog` – View history of branch HEAD changes (lifesaver!)
* Recover lost commits:
  `git checkout <reflog-sha>` – Jump back to previous state

## **Submodules**
* Used when a repo includes another repo
* `git submodule add <repo-url> path/` – Add a submodule
* `git submodule update --init --recursive` – Clone & initialize submodules
* Always run this after cloning a repo with submodules

# Branch naming
## Format:
```
<type>/<short-description>
```

The slash (`/`) convention creates a natural groupings (like folders) in Git tools.

## Prefix Meanings

* `feature/` – New feature
* `bugfix/` – Fix for a bug (non-critical)
* `hotfix/` – Urgent fix for production
* `task/` – Maintenance, cleanup, or non-feature work
* `docs/` – Documentation updates
* `test/` – Experiment or spike

## Description
* Use **kebab-case** (`-`) for readability: `feature/add-user-auth`
* Use git citool to stage and write commits with a clear separation between header and body.
```
feature/add-user-auth

- Created login form and basic validation
- Integrated backend token authentication
- Updated user model to store auth tokens
- This lays the groundwork for future role-based access
```

## Commit Message Format
* Header: One short sentence (max ~72 chars) describing the change
    * Should match your branch name prefix: feature/, bugfix/, etc.
* Body: Bullet points or paragraph explaining:
    * What was done
    * Why it was done
    * Any context, edge cases, limitations

## Integrated With Agile
* Use issue/ticket numbers if your team tracks tasks:
  ```
  <type>/TICKET-ID-Short-Title
  feature/123-add-user-auth
  bugfix/456-header-align
  ```
* No spaces, no uppercase letters, no special characters
* Stick to one naming scheme across the team

# Example Git Workflow

## Starting a New Task
1. `git status`
   *Ensure a clean working directory*
2. `git fetch`
   *Update remote tracking branches*
3. `git checkout -f -B main origin/main`
   *Reset local `main` to latest upstream (safe baseline)*
4. `git branch feature/123-add-user-auth`
   *Create a new branch matching the ticket ID*
5. `git checkout feature/123-add-user-auth`
   *Switch to your new feature branch*
6. Work on your task
7. `git citool`
   *Stage changes and write a detailed commit message*

   ```
   feature/123-add-user-auth

   - Created login form and basic validation
   - Integrated backend token authentication
   - Updated user model to store auth tokens
   - This lays the groundwork for future role-based access
   ```
8. `git push origin feature/123-add-user-auth`
   *Push your branch to GitHub*
9. Create the **Pull Request** on GitHub

## Resolving Review Issues

* Minor fixes can be done directly in GitHub; for anything meaningful, do it locally:

1. `git status`
   *Check for uncommitted changes*
2. `git fetch`
3. `git checkout -f -B feature/123-add-user-auth origin/feature/123-add-user-auth`
   *Ensure local branch matches latest PR version*
4. Make your changes
5. `git citool`
   *Stage and commit the fix*
6. `git push origin feature/123-add-user-auth`

## **Resolving Merge Conflicts**

1. `git status`
2. `git fetch`
3. `git checkout -f -B feature/123-add-user-auth origin/feature/123-add-user-auth`
4. `git merge origin/main`
   *This may trigger a conflict*
5. `git mergetool`
   *Use visual merge tool to resolve conflicts*
6. `git merge --continue`
   *Complete the merge once resolved*
7. `git citool`
   *Commit the resolved merge if needed*
8. `git push origin feature/123-add-user-auth`

## **Integrating Updates from `origin/main` during the workflow**

> Use this when you're working on a task and `origin/main` gets updated — it's best to rebase early before your work diverges too far.

1. `git status`
   *Ensure a clean working directory*
2. `git fetch`
   *Update remote tracking branches*
3. `git checkout -f -B main origin/main`
   *Reset local `main` to latest upstream (safe baseline)*
4. `git branch feature/123-add-user-auth`
   *Create a new branch matching the ticket ID*
5. `git checkout feature/123-add-user-auth`
   *Switch to your new feature branch*
6. Work on your task
7. Need to integrating updates from `origin/main` 
8. `git citool`
   *Stage changes and write a detailed commit message*
9. `git fetch`
   *Update remote tracking branches*
10. `git rebase -i origin/main feature/123-add-user-auth`
   *Rebase your branch onto the updated `main` with an interactive view*
   * This helps review each commit and keep things clean
11. Resolve any conflicts using `git mergetool` (if any)   
12. `git push origin feature/123-add-user-auth`
   *Push your branch to GitHub*
13. Create the **Pull Request** on GitHub
14. Continue working on your task
15. `git citool`
   *Stage changes and write a detailed commit message*
15. `git push origin feature/123-add-user-auth`
   *Push your rebased branch (no force needed since it wasn’t pushed before)*

