# tickets

One file per ticket: `<id>-<track>.md` with frontmatter (`status: open | doing | done | dropped`,
`kind: task | question`, `owner: you | agent`, `depends: [ids]`). Notes and answers go in the body.

Tracks: `app` (UI & search) · `data` (corpus) · `eval` (quality test set).

    grep -H "^status: open" tickets/*.md       # what's open
    grep -l "^owner: you" tickets/*.md          # what's mine

Agents: set `status: doing` when you start, `done` when finished, and write what changed in the body.
