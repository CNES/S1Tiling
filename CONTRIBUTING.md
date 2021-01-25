# How to contribute to S1 Tiling ?

Thank you for taking the time to contribute to S1 Tiling! This document will
guide you through the workflow and best practices you need to know to send your
contribution.

There are many ways to contribute to S1 Tiling:

* [Reporting a bug](#reporting-bugs)
* [Making a feature request](#feature-requests-and-discussions)
* [Contributing code (Python, etc.) or documentation](#code-contribution-and-documentation-improvements)

Our main workflow uses GitLab for source control, issues and task tracking. We
use a self-hosted GitLab instance:

[https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling)

## Reporting bugs

If you have found a bug, you can first [search the existing issues](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/issues?label_name%5B%5D=bug)
to see if it has already been reported.

If it's a new bug, please [open a new issue on GitLab](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/issues/new).
The 'Bug' issue template will help you provide all important information and
help fixing the bug quicker. Remember to add as much information as possible!

## Feature requests and discussions

Feature requests are welcome! Generally you are welcome to simply
[open an issue](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/issues/new)
and discuss your idea there.

## Code contribution and documentation improvements

The S1 Tiling workflow is based on GitLab [Merge Requests](https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html).
Clone the repository, create a feature branch, commit your changes, push the
feature branch to a fork (or the main repository if you are a core developer),
then send a merge request. Direct push to develop without review must be
avoided.

You are also welcome to contribute to the documentation by editing files in docs
folder, commit and sending a merge request.

### Commit message

On your feature branch, write a good [commit message](https://chris.beams.io/posts/git-commit/):
short and descriptive, in imperative style.
If fixing an issue or bug, put the issue number in the commit message so that
GitLab can [cross-link it](https://docs.gitlab.com/ce/user/project/issues/crosslinking_issues.html).
You can prefix your commit message with an indicating flag (DOC, BUG, PKG, TST,
etc.).

Standard prefixes for S1 Tiling commit messages:

    BUG: Fix for runtime crash or incorrect result
    WRN: Fix a warning (from linters...)
    DOC: Documentation change
    ENH: New functionality
    PERF: Performance improvement
    STY: No logic impact (indentation, comments)
    WIP: Work In Progress not ready for merge
    TST: Tests related commits

For example, here are some good commit messages:

    BUG: #1701 Warn users if parameter string is unset
    DOC: Fix typo in installation documentation

### Merge request

Your contribution is ready to be added to the main S1 Tiling repository? Send a
Merge Request against the `develop` branch on GitLab using the merge request
template. The merge request will then be discussed by the community and the
core S1 Tiling team.

* Merge requests can not be merged until all discussions have been resolved
  (this is enforced by GitLab)
* The merger is responsible for checking that the branch is up-to-date with
  `develop`

### Contribution license agreement

S1 Tiling requires that contributors sign out a
[Contributor License Agreement](https://en.wikipedia.org/wiki/Contributor_License_Agreement).
The purpose of this CLA is to ensure that the project has the necessary
ownership or grants of rights over all contributions to allow them to
distribute under the chosen license.

To accept your contribution, we need you to complete, sign and email to *cla [at]
orfeo-toolbox [dot] org* an
[Individual Contributor Licensing Agreement](_static/icla-en.docx) (ICLA) form
and a [Corporate Contributor Licensing Agreement](_static/ccla-en.docx) (CCLA)
form if you are contributing on behalf of your company or another entity which
retains copyright for your contribution.

<!-- NOTE: The two previous links won't work from the markdown file, but
     they'll work once translated into RST
-->

The copyright owner (or owner's agent) must be mentioned in headers of all
modified source files.

## GitLab guidelines

In order to organize the issues in our GitLab instance, we use both labels and
milestones.

The
[milestones](https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/milestones)
should be used to track in which release a feature is merged.  GitLab can then
provide a summary of all features and bugs added to a given release
version.
