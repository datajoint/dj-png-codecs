See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `uvx nox` to run it without installing, or `uv tool install nox`. If
you don't have uv, you can
[install it a variety of ways](https://docs.astral.sh/uv/getting-started/installation/),
including with pip, pipx, brew, and just downloading the binary (single file).

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s docs  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
uv sync
```

# Pre-commit

You should prepare pre-commit or prek, which will help you by checking that
commits pass required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
uv run pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=dj-png-codecs
```

# Building docs

You can build and serve the docs using:

```bash
nox -s docs
```

You can build the docs only with:

```bash
nox -s docs --non-interactive
```

# Releasing

A release is an annotated tag `vX.Y.Z` on `main` whose message is the release
notes. There is no changelog file: `git show vX.Y.Z` prints a release's notes,
and its GitHub release shows the same text.

1. Collect the pull requests merged since the last release (for the first
   release, leave out `--search`):

   ```bash
   gh pr list --state merged --base main --search "merged:>YYYY-MM-DD" --json number,title
   ```

2. Write `notes.md`. The first line is one sentence that says what the release
   is about. Then, after a blank line, the changes grouped by area, each naming
   its pull request. Put anything users must act on first.

   ```markdown
   Bool images are stored as 1-bit PNG files.

   ## Storage

   - `bool` arrays are written as 1-bit grayscale images (#123)
   ```

3. Tag the release commit on `main` and push the tag:

   ```bash
   git tag -a vX.Y.Z --cleanup=verbatim -F notes.md && git push origin vX.Y.Z
   ```

   `--cleanup=verbatim` keeps lines that start with `#`, such as Markdown
   headings, which git would otherwise drop as comments.

Pushing the tag runs the [CD workflow](workflows/cd.yml). It checks that the tag
is annotated and on `main`, and refuses a lightweight tag (one made without
`-a`); runs the CI workflow; publishes the package built at the tag through PyPI
trusted publishing; and creates the GitHub release, with the tag's message as
its notes and the wheel and sdist attached.

The version comes from the tag: hatch-vcs derives it at build time, so there is
no version number to edit, and the workflow checks that the built files carry
the tag's version. If a job fails after the tag is pushed, fix the cause and
re-run the failed jobs from the workflow run; a fix that needs a new commit is
released under the next version. Never move or delete a pushed tag: the version
on PyPI and the GitHub release are tied to it.
