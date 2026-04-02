# Experiment Log

## Structure

One section per method. Within a section, entries are sequential runs — hyperparameter tuning, architectural tweaks, anything that doesn't change the fundamental research question. A new section is warranted when the research question changes (e.g. a different variance reduction mechanism, a different algorithm), not just when the architecture changes.

Entries may branch: a run can have multiple children (alternative directions explored from the same starting point). Record the parent explicitly in each entry. The document is linear but the logical structure is a tree.

## Per-Entry Format

**Before the run:**
- `Parent:` which run this branches from (omit for the first run in a section)
- `Change:` what is different from the parent (one sentence)
- `Hypothesis:` what you expect and why — write this before launching

**After the run:**
- `Tag:` git tag of the commit used (`git tag <section>-NNN-<short-name>`)
- `Run ID:` wandb run ID
- `Result:` the key metrics relevant to the hypothesis
- `Interpretation:` did it match? why or why not?
- `Next:` what this implies to try next

## Discipline

- Write the hypothesis before running. Post-hoc rationalization is the main failure mode.
- Tag the commit before launching. This lets you `git checkout <tag>` to recover the exact code state.
- This file holds reasoning only — wandb holds curves and metrics.

---

<!-- Sections go below -->
