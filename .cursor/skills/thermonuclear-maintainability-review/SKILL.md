---
name: thermonuclear-maintainability-review
description: Run an unusually strict review focused on implementation quality, maintainability, abstraction quality, and codebase health. Also covers giant files and spaghetti-condition growth with ambitious code-judo restructuring. Use for a thermo-nuclear code quality review, thermonuclear review, deep code quality audit, or especially harsh maintainability review.
---

# Thermonuclear Maintainability Review

Use this skill for an unusually strict review focused on implementation quality, maintainability, abstraction quality, and codebase health.

Above all, this skill should push the reviewer to be **ambitious** about code structure. Do not merely identify local cleanup opportunities. Actively search for "code judo" moves: restructurings that preserve behavior while making the implementation dramatically simpler, smaller, more direct, and more elegant.

Default posture: **guilty until proven simple**. Prefer concrete file/line findings over general advice. Do not soften ratings to be polite. Optimize for long-term editability, not cleverness or local elegance.

## When to use

Apply when the user asks for any of:
- thermo-nuclear / thermonuclear code quality review
- deep code quality audit
- especially harsh maintainability review
- unusually strict review of implementation quality / maintainability / abstraction quality / codebase health
- review focused on abstractions, giant files, or condition spaghetti

## Scope first

1. Identify the target: current branch changes, PR diff, paths, or whole repo area.
2. Prefer the changed surface, then follow smells into callees/callers one hop.
3. Skip generated code, vendored deps, and lockfiles unless they contain hand-edited logic.
4. If scope is huge, review the worst offenders first (largest files, deepest branches, newest churn).

## Core Prompt

Start from this baseline:

> Perform a deep code quality audit of the current branch's changes.
> Rethink how to structure / implement the changes to meaningfully improve code quality without impacting behavior.
> Work to improve abstractions, modularity, reduce Spaghetti code, improve succinctness and legibility.
> Be ambitious, if there is a clear path to improving the implementation that involves restructuring some of the codebase, go for it.
> Be extremely thorough and rigorous. Measure twice, cut once.

## Non-Negotiable Additional Standards

Apply the baseline prompt above, plus these explicit review rules:

0. **Be ambitious about structural simplification.**
   - Do not stop at "this could be a bit cleaner."
   - Look for opportunities to reframe the change so that whole branches, helpers, modes, conditionals, or layers disappear entirely.
   - Prefer the solution that makes the code feel inevitable in hindsight.
   - Assume there is often a "code judo" move available: a re-organization that uses the existing architecture more effectively and makes the change dramatically simpler and more elegant.
   - If you see a path to delete complexity rather than rearrange it, push hard for that path.

1. **Do not let a PR push a file from under 1k lines to over 1k lines without a very strong reason.**
   - Treat this as a strong code-quality smell by default.
   - Prefer extracting helpers, subcomponents, modules, or local abstractions instead of letting a file sprawl past 1000 lines.
   - If the diff crosses that threshold, explicitly ask whether the code should be decomposed first.
   - Only waive this if there is a compelling structural reason and the resulting file is still clearly organized.

2. **Do not allow random spaghetti growth in existing code.**
   - Be highly suspicious of new ad-hoc conditionals, scattered special cases, or one-off branches inserted into unrelated flows.
   - If a change adds "weird if statements in random places", treat that as a design problem, not a stylistic nit.
   - Prefer pushing the logic into a dedicated abstraction, helper, state machine, policy object, or separate module instead of tangling an existing path.
   - Call out changes that make the surrounding code harder to reason about, even if they technically work.

3. **Bias toward cleaning the design, not just accepting working code.**
   - If behavior can stay the same while the structure becomes meaningfully cleaner, push for the cleaner version.
   - Do not rubber-stamp "it works" implementations that leave the codebase messier.
   - Strongly prefer simplifications that remove moving pieces altogether over refactors that merely spread the same complexity around.

4. **Prefer direct, boring, maintainable code over hacky or magical code.**
   - Treat brittle, ad-hoc, or "magic" behavior as a code-quality problem.
   - Be skeptical of generic mechanisms that hide simple data-shape assumptions.
   - Flag thin abstractions, identity wrappers, or pass-through helpers that add indirection without buying clarity.

5. **Push hard on type and boundary cleanliness when they affect maintainability.**
   - Question unnecessary optionality, `unknown`, `any`, or cast-heavy code when a clearer type boundary could exist.
   - Prefer explicit typed models or shared contracts over loosely-shaped ad-hoc objects.
   - If a branch relies on silent fallback to paper over an unclear invariant, ask whether the boundary should be made explicit instead.

6. **Keep logic in the canonical layer and reuse existing helpers.**
   - Call out feature logic leaking into shared paths or implementation details leaking through APIs.
   - Prefer existing canonical utilities/helpers over bespoke one-offs.
   - Push code toward the right package, service, or module instead of normalizing architectural drift.

7. **Treat unnecessary sequential orchestration and non-atomic updates as design smells when the cleaner structure is obvious.**
   - If independent work is serialized for no good reason, ask whether the flow should run in parallel instead.
   - If related updates can leave state half-applied, push for a more atomic structure.
   - Do not over-index on micro-optimizations, but do flag avoidable orchestration complexity that makes the implementation more brittle.

## Primary Review Questions

For every meaningful change, ask:

- Is there a "code judo" move that would make this dramatically simpler?
- Can this change be reframed so fewer concepts, branches, or helper layers are needed?
- Does this improve or worsen the local architecture?
- Did the diff add branching complexity where a better abstraction should exist?
- Did a previously cohesive module become more coupled, more stateful, or harder to scan?
- Is this logic living in the right file and layer?
- Did this change enlarge a file or component past a healthy size boundary?
- Are there repeated conditionals that signal a missing model or missing helper?
- Is the implementation direct and legible, or does it rely on special cases and incidental control flow?
- Is this abstraction actually earning its keep, or is it just a wrapper?
- Did the diff introduce casts, optionality, or ad-hoc object shapes that obscure the real invariant?
- Is this logic living in the canonical layer, or did the diff leak details across a boundary?
- Is this orchestration more sequential or less atomic than it needs to be?

## What to Flag Aggressively

Escalate findings when you see:

- A complicated implementation where a cleaner reframing could delete whole categories of complexity.
- Refactors that move code around but fail to reduce the number of concepts a reader must hold in their head.
- A file crossing 1000 lines due to the PR, especially if the new code could be split out.
- New conditionals bolted onto unrelated code paths.
- One-off booleans, nullable modes, or flags that complicate existing control flow.
- Feature-specific logic leaking into general-purpose modules.
- Generic "magic" handling that hides simple structure and makes the code harder to reason about.
- Thin wrappers or identity abstractions that add indirection without simplifying anything.
- Unnecessary casts, `any`, `unknown`, or optional params that muddy the real contract.
- Copy-pasted logic instead of extracted helpers.
- Narrow edge-case handling implemented in the middle of an already busy function.
- Refactors that technically pass tests but make the code less modular or less readable.
- "Temporary" branching that is likely to become permanent debt.
- Bespoke helpers where the codebase already has a canonical utility for the job.
- Logic added in the wrong layer/package when it should live somewhere more central.
- Sequential async flow where obviously independent work could stay simpler and clearer with parallel execution.
- Partial-update logic that leaves state less atomic than necessary.

## Preferred Remedies

When you identify a code-quality problem, prefer suggestions like:

- Delete a whole layer of indirection rather than polishing it.
- Reframe the state model so conditionals disappear instead of getting centralized.
- Change the ownership boundary so the feature becomes a natural extension of an existing abstraction.
- Turn special-case logic into a simpler default flow with fewer exceptions.
- Extract a helper or pure function.
- Split a large file into smaller focused modules.
- Move feature-specific logic behind a dedicated abstraction.
- Replace condition chains with a typed model or explicit dispatcher.
- Separate orchestration from business logic.
- Collapse duplicate branches into a single clearer flow.
- Delete wrappers that do not meaningfully clarify the API.
- Reuse the existing canonical helper instead of introducing a near-duplicate.
- Make type boundaries more explicit so the control flow gets simpler.
- Move the logic to the package/module/layer that already owns the concept.
- Parallelize independent work when that also simplifies the orchestration.
- Restructure related updates into a more atomic flow when partial state would be harder to reason about.

Do not be satisfied with "maybe rename this" feedback when the real issue is structural.
Do not be satisfied with a merely cleaner version of the same messy idea if there is a plausible path to a much simpler idea.

## Review pillars (deep dive)

Use these pillars to pressure-test the non-negotiable standards above.

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

Hard rule from the non-negotiable standards: do not let a PR push a file from under 1k lines to over 1k lines without a very strong reason.

Additional heuristics (adjust only if the repo already has clearer norms):
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

Control-flow entropy is a primary review target.

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
3. Actively search for code-judo reframes that delete branches, helpers, modes, or layers while preserving behavior.
4. Score only from evidence in code, not intent.
5. Separate **blocking maintainability defects** from **follow-ups**.

## Review Tone

Be direct, serious, and demanding about quality.
Do not be rude, but do not soften major maintainability issues into mild suggestions.
If the code is making the codebase messier, say so clearly.
If the implementation missed an opportunity for a dramatic simplification, say that clearly too.

Good phrases:

- `this pushes the file past 1k lines. can we decompose this first?`
- `this adds another special-case branch into an already busy flow. can we move this behind its own abstraction?`
- `this works, but it makes the surrounding code more spaghetti. let's keep the behavior and restructure the implementation.`
- `this feels like feature logic leaking into a shared path. can we isolate it?`
- `this abstraction seems unnecessary. can we just keep the direct flow?`
- `why does this need a cast / optional here? can we make the boundary more explicit instead?`
- `this looks like a bespoke helper for something we already have elsewhere. can we reuse the canonical one?`
- `i think there's a code-judo move here that makes this much simpler. can we reframe this so these branches disappear?`
- `this refactor moves complexity around, but doesn't really delete it. is there a way to make the model itself simpler?`

Blunt, specific, actionable. No filler praise. No personal attacks. Roast the design, not the author.

## Output Expectations

Prioritize findings in this order:

1. Structural code-quality regressions
2. Missed opportunities for dramatic simplification / code-judo restructuring
3. Spaghetti / branching complexity increases
4. Boundary / abstraction / type-contract problems that make the code harder to reason about
5. File-size and decomposition concerns
6. Modularity and abstraction issues
7. Legibility and maintainability concerns

Do not flood the review with low-value nits if there are larger structural issues.
Prefer a smaller number of high-conviction comments over a long list of cosmetic notes.

## Output format

Use this structure:

```markdown
# Thermonuclear Maintainability Review

## Verdict
[one blunt sentence: ship / ship-with-guards / blocked for maintainability]
[explicitly state whether the approval bar is met]

## Severity Legend
- P0: will actively harm every future change in this area / presumptive blocker
- P1: high drag; fix before expanding feature surface
- P2: real smell; schedule soon
- P3: nit / polish

## Top Findings
### P0 — [title]
- Where: `path/to/file` (lines X–Y) and related `path`
- Smell: code-judo-miss | structural-regression | abstraction | giant-file | condition-spaghetti | boundary/type | orchestration | other
- Why it hurts: [specific future edit that becomes dangerous]
- Evidence: [concrete symbols/branches/sizes]
- Blast radius: [what else is entangled]
- Fix: [specific refactor steps / target decomposition / code-judo reframe]

## File Hotspots
| File | Lines (approx) | Crossed 1k via PR? | Responsibilities | Recommendation |
|------|----------------|--------------------|------------------|----------------|
| ... | ... | yes/no | ... | ... |

## Condition Hotspots
| Location | Nesting / boolean density | Better shape |
|----------|---------------------------|--------------|
| ... | ... | ... |

## Abstraction Scorecard
| Abstraction | Earns its keep? | Problem | Action |
|-------------|-----------------|---------|--------|
| ... | yes/no | ... | inline/split/rename/redesign/delete |

## Change-Scenario Stress Test
Pick 2–3 likely edits and show why the current design makes them fragile.
Call out any code-judo reframes that would make those edits dramatically simpler.

## What Passes
Only list genuinely solid parts. Do not pad.

## Refactor Sequence
Ordered steps, smallest valuable wedge first. No vague "clean up later".
Prefer steps that delete complexity, not merely rearrange it.
```

## Scoring rules

- If a file is huge **and** owns tangled conditions, escalate at least one severity.
- If an abstraction exists mainly to paper over the tangle, mark the abstraction as a finding, not a virtue.
- If there is a plausible code-judo move that deletes complexity and it was missed, escalate.
- Do not trade readability for pattern cosplay (factories, DI graphs, enterprise layers) unless they remove real branching/ownership pain.
- "It works" is not a defense.

## Approval Bar

Do not approve merely because behavior seems correct.
The bar for approval is:

- no clear structural regression
- no obvious missed opportunity to make the implementation dramatically simpler when such a path is visible
- no unjustified file-size explosion
- no obvious spaghetti-growth from special-case branching
- no obviously hacky or magical abstraction that makes the code harder to reason about
- no unnecessary wrapper/cast/optionality churn obscuring the real design
- no clear architecture-boundary leak or avoidable canonical-helper duplication
- no missed opportunity for an obvious decomposition that would materially improve maintainability

Treat these as presumptive blockers unless the author can justify them clearly:

- the PR preserves a lot of incidental complexity when there is a plausible code-judo move that would delete it
- the PR pushes a file from below 1000 lines to above 1000 lines
- the PR adds ad-hoc branching that makes an existing flow more tangled
- the PR solves a local problem by scattering feature checks across shared code
- the PR adds an unnecessary abstraction, wrapper, or cast-heavy contract that makes the design more indirect
- the PR duplicates an existing helper or puts logic in the wrong layer when there is a clear canonical home

If those conditions are not met, leave explicit, actionable feedback and push for a cleaner decomposition.
