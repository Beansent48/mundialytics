# User accounts: design (2026-10-21)

Tasks **B1** and **B2** of the [platform roadmap](PLATFORM_ROADMAP.md). B1 is
the foundation — a user can log in and the app knows who they are. B2 is what
the account is *for* — favourites, settings and SquadLab progress stop living in
the browser.

Decided with the user on 2026-10-21: **public site, optional accounts**; login
with **Google + email through a library**, users in **our own Postgres**, no
paid managed service; and the account eventually carries **achievements,
objectives and user levels over SquadLab**.

---

## 1. The principle: nothing gets locked

Every page that works today keeps working for a visitor who never logs in. The
account unlocks *saving*, not *seeing*. Two consequences the implementation has
to respect:

- No route may become auth-only. Signing in is an affordance in the header, not
  a gate in `web/proxy.ts`.
- Anything an anonymous visitor can do today must still work from
  `localStorage` after accounts exist, because that is what a logged-out
  visitor will keep using. The storage layer gets two backends, not a
  replacement.

## 2. Where authentication lives: Next.js, not FastAPI

**The Python API stays stateless and anonymous.** Accounts live entirely in the
Next.js app: Auth.js/Better Auth handlers under `web/app/api/auth/...`, user
data read and written by Next route handlers (or server actions) against
Postgres, and the FastAPI in `api/` continues to answer the same public
football questions it answers today.

Why, explicitly:

- `api/` is 3,000 lines of modelling service with a joblib fit cache and no
  notion of a request identity. Adding sessions, password hashing and a user
  table to it would double its surface and couple the model server to the
  product's account system.
- Session cookies are a browser concern and the browser already talks to Next.
  A token that has to be minted in Next, forwarded to FastAPI and verified
  there is a second auth implementation for no gain.
- It keeps the deployment story simple in **A2**: the API container needs no
  database at all.

The one thing to watch: the web currently calls the API from the server with a
shared `next: { revalidate, tags: ["data"] }` cache. **Per-user responses must
never go through that path**, or one user's favourites get cached and served to
another. Rule: public football data goes through `web/lib/api.ts` and is
cached; user data goes through its own module, is never tagged `data`, and is
fetched with `cache: "no-store"`.

## 3. Stack, and the trap in it

- **Postgres**, in Docker locally (one `postgres` service in `docker-compose`
  with a named volume) and on the VPS later. One engine, both places.
- **An auth library, not hand-rolled**: Better Auth or Auth.js v5. B1 picks one
  after reading the *current* docs — both moved fast in 2026 — with a bias to
  Better Auth for first-class Drizzle support and email+password without a
  separate adapter.
- **A typed query layer**: Drizzle, with migrations committed to the repo. The
  project's existing discipline — a schema change is a reviewable diff — applies
  to the database too.
- **Secrets** in `web/.env.local`, never committed: the Google OAuth client, the
  session secret, `DATABASE_URL`. A committed `.env.example` lists every key
  with no value.

> **The trap.** Next.js 16 renamed `middleware.ts` to **`proxy.ts`**, and this
> project has one already (`web/proxy.ts`, locale routing, with a hard-won
> comment about matcher semantics). Every auth guide written before 2026 says
> "add `middleware.ts`". Doing that creates a file Next will not run, and the
> session guard silently never fires. Auth checks go *into the existing*
> `proxy.ts`, after the locale rewrite, or into the route handlers themselves.

## 4. The schema

Sketch, not gospel; B1 writes the auth half and B2 the app half.

**From the auth library** (B1) — shape dictated by whichever is chosen:
`user`, `account` (one row per OAuth provider link), `session`, `verification`.

**Ours** (B2):

```
user_settings      user_id PK, locale, theme, odds_format, created_at, updated_at
user_favorites     user_id, kind (team | competition), value, created_at
                   PK (user_id, kind, value)
squadlab_run       id, user_id, mode (draft | historic), seed,
                   squad (jsonb: the chosen names), rerolls (jsonb),
                   status (in_progress | finished), result (jsonb, nullable),
                   engine_fingerprint, created_at, updated_at
user_progress      user_id PK, xp, level, updated_at          -- gamification
achievement        code PK, name, description, points         -- seeded, not user data
user_achievement   user_id, code, awarded_at, PK (user_id, code)
```

Two things that make this smaller than it looks:

- **`user_favorites` already exists in the client's shape.**
  `web/lib/favorites.ts` stores `{teams, competitions}` and its docstring says
  it is "a deliberate staging post" for exactly this. The component API
  (`useFavorites()` returning `{favorites, toggle, has, ready}`) does not
  change; only what sits behind it does.
- **A SquadLab run is reproducible from its inputs.** `api/squadlab.py` is
  seed-driven throughout: `deal_slot(seed, position, index, reroll)` and
  `play_champions(squad_names, seed)`. A saved run is a seed, a list of names
  and the rerolls — not a stored season. Store `result` as well *only* because
  ratings and pools move with the data: a run replayed after a refresh can come
  out different, so a finished run keeps its outcome and its
  `engine_fingerprint` and is never silently re-simulated. That is the same
  reasoning the pre-kickoff prediction log already uses.

## 5. Migration from `localStorage`

First login is the only interesting moment:

1. Read the browser's `mv.favorites.v1` (and the SquadLab state, once B2 stores
   any).
2. **Merge, never replace** — union of teams and competitions. Someone who
   followed Valencia on their phone and Betis on their laptop ends with both.
3. Write the merged set server-side, then mark the local copy migrated (keep it
   as a cache for offline reads; do not delete it, so logging out is not data
   loss).
4. From then on: logged in, the server is the truth with the local copy as an
   optimistic cache; logged out, local only, exactly as today.

## 6. Gamification, and why it shapes B2

Achievements, objectives and levels are explicitly wanted, and explicitly
later. What B2 owes them is only this: **every SquadLab run is a durable row
with a status and a result**, so an award rule can be evaluated against history
("won the Champions with no card above 80", "three finals in a row") instead of
needing the runs replayed. Points are then a pure function over `squadlab_run`
and `user_achievement`, and the game design can land without a migration.

What B2 must *not* do: invent achievement codes, XP curves or level thresholds.
Empty tables and a documented shape; the rules are their own task.

## 7. Privacy, since this will be public

Small, but not optional:

- Minimum data: an email, a display name and an avatar URL from Google. No
  birthdays, no phone numbers, nothing a feature does not use.
- **Delete account has to really delete** — the user row and everything keyed
  on `user_id`. A cascade in the schema, plus a test.
- A session cookie that is strictly necessary needs no consent banner; the
  moment anything analytics-shaped is added, that changes. Do not add it
  casually.
- A privacy page, with its strings in `web/messages/{en,es}.json` like every
  other string. The site is bilingual and the account pages must be too.

---

## Open questions, for when B2 starts

Not blocking — each has a sane default, recorded here so the choice is
conscious rather than accidental.

| Question | Default if nobody decides |
|---|---|
| One SquadLab run at a time, or many saved runs? | Many: `squadlab_run` rows with a status, and the UI offers "continue" for the one in progress. |
| Can a logged-out visitor still play SquadLab? | Yes, in the browser, and the run is offered for saving on login. |
| Do settings (locale, theme) follow the account across devices? | Yes — that is most of the point. The cookie stays authoritative for the current request. |
| Usernames, or just the Google display name? | Display name. A unique-handle system is only worth it once something is public, like a leaderboard. |
