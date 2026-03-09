---
name: review-changes
description: Review currently uncommitted changes
context: fork
argument-hint: goal of changes
---

Someone, either you or a human programmer, has made changes to the codebase. They are intended to
accomplish the following:

$ARGUMENTS

To review:
* Identify all changed and new files using `git status`.
* Thoroughly read the changes, and enough context to understand them fully. For this review,
  consider only code and configuration changes that are relevant to the goal.
* Thoroughly review them for compliance with the code style guidelines.
* Thoroughly review them for correctness.
* Thoroughly review whether they accomplish the goal.
* Report any issues in a prioritized list. Also report any changes that you chose NOT to review
  because they weren't relevant to the goal.