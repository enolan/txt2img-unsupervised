---
name: autonomous-pr-workflow
description: Write code and publish a PR autonomously from end-to-end. Use this when given a complete task description (a bug to fix, feature to implement, etc) and asked to submit a PR when done.
user-invocable: false
---

If you're using this skill, you're doing some programming work with the goal of producing a
reviewable and ideally ready-to-merge PR. You should assume by default that no human is actually
looking at your session and work autonomously. Imagine you're a senior engineer and your supervisor
has asked you to do this task. If you can figure out how to accomplish it on your own, work until
you've got a finished PR, but if in your professional judgement you need to talk to me in order to
continue, or a conversation would save us a lot of time and tokens, then contact me. Any time you
need to do so, use the PushNotificationTool to ping my phone. That will draw my attention, and I'll
look at what you've said in the conversation then.

Your goal is a GitHub pull request, against the enolan/txt2img-unsupervised repo, that is correct,
tested, high quality, and easy to review. Specifically:

* **Use however many commits makes the PR easiest to review.** Use a single commit for simpler PRs,
  and two or more if you can logically split the PR into pieces and each piece makes sense and is
  reviewable independently. For instance, if you need to first do a refactor in order to add a new
  feature, that is likely better as two commits than one.
* **Iterate with the review-changes skill before submitting.** Before committing, run the skill to
  find issues, review its output, and fix the problems it reports **if** you agree they are real
  problems and fixing them would be an improvement to the PR. You'll have to use your judgement. If
  you fix issues the skill identifies, rerun it afterward to search for new issues. Iterate until
  there are no more valid issues, or you've gone around the loop three times.
* **Test your changes before submitting.** Use whatever combination of automated (e.g. pytest) and
  manual (e.g. training a toy model) tests make sense for the work you're doing. You shouldn't
  submit anything until you're confident it's correct.
* **Write useful commit messages.** The idea here, as always, is to make it easy to review your
  work, and to understand it in the future. The subject line should be a brief, scannable
  description of what was done and why. The body can be longer and go into more depth, but shouldn't
  merely duplicate what you'd get by reading the code. It's primarily for context on the *change*,
  whereas descriptions of the code as it stands should go in comments. For example "we changed from
  algorithm X to Y because in context Z X has pathological memory blowup" is a commit-message type
  thing, whereas "here we need an algorithm for problem A. we have reqs B, C, & D, and choose algo Y
  which is a pretty good tradeoff here though we may need to change it to something else if
  situation Q happens" is a code comment.
* **Write useful PR descriptions.** Your PR description should recapitulate what I asked you to do,
  how you did it, any non-obvious decisions or judgement calls you made, and anything else I need to
  know that isn't contained in the commit messages or code. Also describe how you tested the changes
  and why that convinces you the code is correct. Note that commits get saved in the Git history and
  PR descriptions don't, text useful for someone digging through history should go in commit
  messages while stuff that's primarily useful only to the reviewer should live in PR descriptions.