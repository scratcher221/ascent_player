---
name: thermonuclear-maintainability-review
description: Run an extremely strict maintainability review for abstraction quality, giant files, and spaghetti-condition growth. Use for a thermo-nuclear code quality review, thermonuclear review, deep code quality audit, or especially harsh maintainability review.
---

# Thermonuclear Maintainability Review

Run an adversarial maintainability audit. Assume the code will be owned by strangers under deadline pressure. Optimize for long-term editability, not cleverness or local elegance.

Default posture: **guilty until proven simple**. Prefer concrete file/line findings over general advice. Do not soften ratings to be polite.

## When to use

Apply when the user asks for any of:
- thermo-nuclear / thermonuclear code quality review
- deep code quality audit
- especially harsh maintainability review
- review focused on abstractions, giant files, or condition spaghetti

## Scope first

1. Identify the target: PR diff, paths, or whole repo area.
2. Prefer the changed surface, then follow smells into callees/callers one hop.
3. Skip generated code, vendored deps, and lockfiles unless they contain hand-edited logic.
4. If scope is huge, review the worst offenders first (largest files, deepest branches, newest churn).

## Review pillars

### 1. Abstraction quality

Fail abstractions that do not earn their indirection.

Flag:
- Abstractions with one implementation and no realistic second case
- Indirection that renames concepts without reducing complexity ("Manager", "Handler", "Helper", "Util", "Service" dumping grounds)
- Leaky abstractions that force callers to know internals
- Wrong-layer mixing: I/O inside pure logic, UI inside domain, persistence inside parsers
- Premature generalization: config/flags/interfaces added "just in case"
- Dual sources of truth / parallel hierarchies that must stay in sync
- Anemic wrappers that only delegate
- God objects that orchestrate everything

Ask of every abstraction:
- What decision does it hide?
- What change becomes cheaper because it exists?
- Can a new teammate predict where to edit?

If the answer is weak, recommend **inline, split by responsibility, or rename to the real domain concept**.

### 2. Giant files

Large files are a merge-conflict and comprehension hazard. Be harsh.

Heuristic thresholds (adjust only if the repo already has clearer norms):
- **>300 lines**: suspect; justify or split
- **>500 lines**: serious defect unless it is dumb data
- **>1000 lines**: critical; must propose a decomposition

Flag:
- Files with many unrelated responsibilities
- Mixed feature areas that change for different reasons
- Deeply nested types/classes that should be modules
- "Catch-all" files: `utils`, `helpers`, `common`, `misc`, `old`
- Tests that copy production spaghetti instead of constraining it

When flagging, propose a **concrete split** (target filenames + what goes where), not "consider refactoring".

### 3. Spaghetti-condition growth

Control-flow entropy is the main review target.

Flag:
- Boolean soup: many `&&` / `||` without named predicates
- Nested `if`/`else`/`switch`/`match` ladders (especially >3 levels)
- Special-case accretion: comments like "legacy", "temp", "hotfix", "edge case" guarding permanent branches
- Feature-flag archaeology and dead mode combinations
- State checks scattered across files instead of one state model
- Exception handling used for normal control flow
- Null/optional checks that exist because ownership is unclear
- Duplicated conditionals with slight drift
- Implicit priority/order dependence between distant branches

Prefer recommendations in this order:
1. **Named predicates / policy functions**
2. **Lookup tables / maps / data-driven rules**
3. **Polymorphism or strategy** when variants are stable
4. **State machine / explicit state type** when transitions matter
5. **Split paths early** (guard clauses) to remove nesting

Do not recommend a new framework when a local redesign of conditionals suffices.

## Additional maintainability landmines

Still report when clearly present:
- Hidden global/mutable singletons
- Temporal coupling (methods that must be called in an undocumented order)
- Stringly-typed contracts and magic literals at boundaries
- Comments explaining what code cannot say
- TODOs that encode unfinished architecture
- Copy-paste clusters about to diverge

## Method

For each target area:

1. Inventory the largest files and densest conditional hotspots.
2. Trace one real change scenario ("add a variant", "fix a bug", "remove a flag") and note how many files/branches it touches.
3. Score only from evidence in code, not intent.
4. Separate **blocking maintainability defects** from **follow-ups**.

## Output format

Use this structure:

```markdown
# Thermonuclear Maintainability Review

## Verdict
[one blunt sentence: ship / ship-with-guards / blocked for maintainability]

## Severity Legend
- P0: will actively harm every future change in this area
- P1: high drag; fix before expanding feature surface
- P2: real smell; schedule soon
- P3: nit / polish

## Top Findings
### P0 — [title]
- Where: `path/to/file` (lines X–Y) and related `path`
- Smell: abstraction | giant-file | condition-spaghetti | other
- Why it hurts: [specific future edit that becomes dangerous]
- Evidence: [concrete symbols/branches/sizes]
- Blast radius: [what else is entangled]
- Fix: [specific refactor steps / target decomposition]

## File Hotspots
| File | Lines (approx) | Responsibilities | Recommendation |
|------|----------------|------------------|----------------|
| ... | ... | ... | ... |

## Condition Hotspots
| Location | Nesting / boolean density | Better shape |
|----------|---------------------------|--------------|
| ... | ... | ... |

## Abstraction Scorecard
| Abstraction | Earns its keep? | Problem | Action |
|-------------|-----------------|---------|--------|
| ... | yes/no | ... | inline/split/rename/redesign |

## Change-Scenario Stress Test
Pick 2–3 likely edits and show why the current design makes them fragile.

## What Passes
Only list genuinely solid parts. Do not pad.

## Refactor Sequence
Ordered steps, smallest valuable wedge first. No vague "clean up later".
```

## Scoring rules

- If a file is huge **and** owns tangled conditions, escalate at least one severity.
- If an abstraction exists mainly to paper over the tangle, mark the abstraction as a finding, not a virtue.
- Do not trade readability for pattern cosplay (factories, DI graphs, enterprise layers) unless they remove real branching/ownership pain.
- "It works" is not a defense.

## Tone

Blunt, specific, actionable. No filler praise. No personal attacks. Roast the design, not the author.
