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
  separate adapter. **B1 chose Better Auth** (1.7.12); the three facts that
  decided it are in [§8](#8-what-b1-actually-built).
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

## 8. What B1 actually built

Landed 2026-10-23. Sections 1-3 above were the plan; this is the record.

### The library: Better Auth 1.7.12

Three facts, checked against npm and the libraries' own current docs rather
than against guides:

1. **Auth.js v5 is still beta.** `next-auth` publishes `5.0.0-beta.32` under the
   `beta` tag and `4.24.15` under `latest`. Better Auth publishes `1.7.12` as
   `latest`. Betting the account layer of a public site on a package whose
   maintainers have not called it stable in three years is a cost with no
   matching benefit.
2. **Email and password are core, not an exercise.** Auth.js leaves password
   hashing to the application and its Credentials provider cannot use database
   sessions — which would have meant JWT sessions for email users and database
   sessions for Google users, two session models in one app. Better Auth hashes,
   verifies, issues the verification token and stores every session in the same
   table.
3. **It generates the schema it expects.** `npm --prefix web run db:schema`
   writes `web/lib/db/schema.ts` from the auth config, so the four tables are a
   reviewable diff and not a hand-copy of a documentation page that could drift.

Cost of the choice, recorded honestly: Better Auth is younger, so the blast
radius of a breaking release is ours to absorb, and the version is pinned in
`web/package-lock.json` rather than floating.

### The schema, as built

Four tables, generated from the auth config and committed:

```
user           id (text PK), name, email (unique), email_verified,
               image, created_at, updated_at
session        id, token (unique), expires_at, ip_address, user_agent,
               user_id -> user.id ON DELETE CASCADE
account        id, account_id, provider_id, user_id -> user.id ON DELETE
               CASCADE, access_token, refresh_token, id_token, scope,
               password, …expiry columns
verification   id, identifier, value, expires_at
```

`provider_id` is `credential` for an email and password, `google` for Google.
One person with both has one `user` row and two `account` rows —
`accountLinking` is on, with Google trusted, because Google verifies its own
addresses.

Two properties of this matter to **B2**, which hangs favourites, settings,
SquadLab runs and later the achievements and XP off the same user:

- **`user.id` is `text`.** Every table B2 adds keys on
  `text("user_id").references(() => user.id, { onDelete: "cascade" })`, the
  shape `session` and `account` already use. There is no serial to migrate to
  and no second identity to reconcile.
- **Deletes cascade, and that is tested.**
  `web/tests/account-cascade.test.mjs` asserts both the behaviour and the
  constraint: every foreign key in the schema must carry `delete_rule =
  CASCADE`, so a B2 table that forgets it fails the test rather than leaving
  orphans behind a deleted account.

B1 deliberately added none of B2's tables. Empty tables would be a guess at
columns that B2 will design with the features in front of it, and the design
above assigns them there. What B1 owed the gamification work is the anchor —
one Postgres, one `user` table with a stable text id, cascade by default, and a
migration tool that can add a table — and that exists.

### Where the pieces live

```
docker-compose.yml                  postgres:17-alpine, named volume
web/.env.example                    every key, no values
web/drizzle.config.ts               migration tool config
web/lib/db/client.ts                lazy pg pool + Drizzle
web/lib/db/schema.ts                the four tables (generated)
web/lib/db/migrations/              committed SQL + drizzle's journal
web/lib/auth.ts                     the Better Auth instance
web/lib/auth-config.ts              is Google configured?
web/lib/auth-client.ts              the browser client
web/lib/auth-errors.ts              error codes -> message keys
web/lib/mail.ts                     the one seam a mail transport plugs into
web/app/api/auth/[...all]/route.ts  every auth endpoint
web/app/[locale]/sign-in|sign-up/   the two prerendered forms
web/app/[locale]/account/           the one page rendered per request
web/components/account/             the form, sign-out, delete
```

### Three things that are not obvious from the code

- **`web/proxy.ts` is untouched.** Next.js 16's own documentation says Proxy
  "should not be used as a full session management or authorization solution",
  and nothing on this site is gated anyway, so there is no auth check in it.
  The trap §3 warned about was avoided by not needing the file.
- **The session is read in the browser for the header, on the server for the
  account page.** The header sits in the locale layout above every football
  page, and those pages are prerendered over the locales; a server-side session
  read there would make the entire site render per request to put one name in a
  dropdown. So `AccountMenu` calls `useSession()` and ships its signed-out
  shell in the static HTML. `/account` is the only route with
  `dynamic = "force-dynamic"`.
- **There is no mail transport.** Verification links are written to the server
  log by `web/lib/mail.ts`, which is a real flow for one operator on their own
  machine and costs nothing. `AUTH_REQUIRE_EMAIL_VERIFICATION` therefore
  defaults to off: turning it on without a transport would lock every new
  account out behind a link only the terminal can see. Match alerts (the
  roadmap backlog) are what should bring a real sender, and when they do they
  replace the body of one function.

### Exercised against a real database, 2026-10-23

Not "it compiles". The whole flow was driven through the production build
against the compose Postgres, and these are the things that were actually
observed:

- `docker compose up -d --wait postgres` reports **healthy**, and
  `npm --prefix web run db:migrate` creates the four tables. The cascade rules
  are in `information_schema`: `account` and `session` both `CASCADE`.
- Creating an account writes one `user` row (`email_verified = f`), one
  `account` row with `provider_id = credential` and a non-null hashed password,
  and one `session` row — then lands on `/es/account` with the locale intact.
- The verification link really is a link. `web/lib/mail.ts` printed
  `[mail] to=… url=http://localhost:3000/api/auth/verify-email?token=…`;
  following it returned 302 and flipped `email_verified` to `t`.
- **The session survives more than a reload.** It survived `location.reload()`,
  and it survived the server being stopped, `.next` deleted, rebuilt from
  scratch and restarted — the account page still rendered the signed-in state,
  which is what a cookie in Postgres rather than in memory buys.
- A wrong password returns `INVALID_EMAIL_OR_PASSWORD` and the UI shows *our*
  Spanish sentence, so `lib/auth-errors.ts` is mapping real codes and not
  guesses. The right password signs back in.
- Signing out deletes that session row and leaves `/api/auth/get-session`
  answering `null`.
- **Delete account deletes.** With one user, two sessions and one account in
  the database, confirming with the password left **0 / 0 / 0** — the cascade
  took a second, orphaned session belonging to that user along with it.
  `node --test web/tests/account-cascade.test.mjs` passes against the live
  database; both web test files together are 5 of 5.

Still unexercised: **Google**, which needs an OAuth client only the account
owner can create (§9). Everything up to Google's own door is in place — the
button appears exactly when the two variables are set, and the callback path is
registered by the handler.

### Still open after B1

- **Password reset.** Someone who signs up with an address and forgets the
  password has no way back in. Better Auth ships the flow; it needs the mail
  transport above, so it waits for the same task.
- **Delete, for a Google account signed in long ago.** Better Auth requires a
  password or a session fresher than a day. An email account always has the
  password; a Google account older than that gets told to sign in again, which
  is true but is one click more than it should be. The clean fix is
  `sendDeleteAccountVerification`, which again needs mail.
- **A privacy page**, per §7. Nothing analytics-shaped was added and the
  session cookie is strictly necessary, so there is nothing to disclose yet
  beyond what the account stores — but the page should exist before the site is
  public.

---

## 9. Setting up Google sign-in

The OAuth client has to be created by the account owner; it cannot be scripted,
and the secret should never pass through a chat. Google renames these console
pages periodically, so the steps name what to look for rather than a menu path,
and the one thing that must be exact is the redirect URI.

1. In the [Google Cloud console](https://console.cloud.google.com/), create a
   project (or pick one) — the name is only ever seen by you.
2. Configure the **OAuth consent screen** / branding for the project: an app
   name, your email as the support contact, and **External** as the user type.
   While the app is in testing, only the accounts listed as test users can sign
   in, so add your own address there.
3. Create an **OAuth 2.0 Client ID** of type **Web application**.
4. Add one **authorized redirect URI**, exactly:

   ```
   http://localhost:3000/api/auth/callback/google
   ```

   Exactly means exactly: scheme, port, case and no trailing slash. A mismatch
   is the `redirect_uri_mismatch` error and nothing else. The path comes from
   Better Auth's handler at `web/app/api/auth/[...all]/route.ts`, and it carries
   no locale because `web/proxy.ts` leaves `/api` out of locale routing.

   Add the deployed origin's version of the same path — `https://<domain>/api/auth/callback/google` — when the site goes live in **A2**. Keep
   the localhost one; a client can hold several.
5. Copy the client ID and secret into `web/.env.local` as `GOOGLE_CLIENT_ID`
   and `GOOGLE_CLIENT_SECRET`, then restart the dev server. Next reads
   `.env.local` at startup, and the Google button is rendered from a server-side
   check of those two variables, so until the restart the button stays hidden —
   which is deliberate: a button with no client id behind it leads to Google's
   own error page.

No scopes need requesting beyond the defaults. Better Auth asks for `openid`,
`email` and `profile`, which is the name, address and avatar URL §7 limits us
to.

---

## 10. What B2 actually built

Landed 2026-10-23, on top of [§8](#8-what-b1-actually-built). Sections 4-6 were
the plan; this is the record, including the two places the plan was wrong about
the code.

### The tables, as built

Six, in `web/lib/db/app-schema.ts` — **not** in `schema.ts`, and that is
load-bearing: `schema.ts` is generated from the auth config and
`npm run db:schema` overwrites it, so a table added there would vanish the next
time anyone regenerated the auth schema, silently, because the generator has no
idea it was not the only author. `drizzle.config.ts` reads both files and
`client.ts` hands both to Drizzle.

```
user_settings     user_id PK, locale, theme, odds_format, created_at, updated_at
user_favorites    user_id, kind ('team'|'competition'), value, created_at
                  PK (user_id, kind, value)
squadlab_run      id PK, user_id, mode ('draft'|'sandbox'|'historic'),
                  seed bigint, squad jsonb, rerolls jsonb,
                  status ('in_progress'|'finished'), result jsonb,
                  engine_fingerprint, created_at, updated_at
user_progress     user_id PK, xp, level, updated_at          -- empty
achievement       code PK, name, description, points          -- empty
user_achievement  user_id, code, awarded_at, PK (user_id, code) -- empty
```

Everything cascades from `user.id`, so `tests/account-cascade.test.mjs` — which
insists that *every* foreign key in the schema cascades — passes with eight
foreign keys where B1 had two. The three gamification tables exist and are
written by nothing, exactly as §6 asked: the game design is its own task, and
it should not have to pay for a migration.

Two check constraints do work that application code would otherwise have to
remember: `mode` and `status` are closed sets, and a run that is `finished`
must have a `result`, so "finished but empty" is unrepresentable rather than
merely discouraged.

### Two things the code knew and this document did not

- **SquadLab has three modes, not two.** `squad-lab.tsx` has had a `sandbox`
  mode all along, where you pick freely instead of being dealt cards. §4's
  `mode (draft | historic)` would have become a check constraint that rejected
  every sandbox run at the database. Caught before the migration shipped, which
  is the only reason it is a footnote rather than an incident.
- **A name does not identify a card.** §4 says to store "the chosen names", and
  that is what `play_champions` consumes — but the pool holds the same
  footballer as an "actual", a "prime" and sometimes an "icono", so a resumed
  draft would show a card the person never held. The whole card is stored
  instead. Eighteen small objects, and resume is exact.

The lesson both share: a schema written from a design document and never read
back against the code it has to serve will encode the document's assumptions,
and constraints make those assumptions load-bearing.

### How the data gets to the browser

`lib/user-data.ts` for the queries, `lib/user-route.ts` for the one place a
request becomes a user id, and five route handlers under `app/api/user/`.

**Nothing goes through `lib/api.ts`.** That client caches every call under a
shared `tags: ["data"]`, which is right for public football numbers and would
be a disclosure bug here — one person's favourites cached and handed to the
next. The handlers are `force-dynamic`, every response carries
`cache-control: no-store, private`, and the browser fetches with
`cache: "no-store"`. The test asserts those headers, so the rule has a tripwire
rather than only a paragraph.

Ownership is a `WHERE` clause and never a check after the fact. Somebody else's
run is a **404, not a 403**, because 403 confirms it exists. The upsert carries
the user id in its own `WHERE` too, so a guessed primary key cannot overwrite
another account's row.

### Favourites: two backends, not a replacement

Signed out, `localStorage`, behaving exactly as before accounts existed — no
page was ever gated and a visitor who never signs in must lose nothing. Signed
in, the server is the truth and the local copy becomes the optimistic cache.
`useFavorites()` still returns `{favorites, toggle, has, ready}`; no component
changed, which is what the shape in that file was always for.

First sign-in on a browser is a **union**, per §5. Follow Valencia on a phone
and Betis on a laptop, sign in on the laptop, keep both. The local copy is
never deleted, so signing out is not data loss. A toggle paints before it is
written — a follow button that waits for a round trip feels broken even when
the round trip is fast — but adopts what the server says back, so a failed
write shows the truth rather than a heart that was never saved.

### Settings: theme and language are not symmetrical

Theme is applied silently on sign-in; nobody is surprised that dark mode
followed them. **Language is applied only when this device has expressed no
opinion** — no `NEXT_LOCALE` cookie. Re-rendering the page someone is reading
into another language is the kind of helpful that feels like a bug, and
next-intl resolves the locale from the URL and the cookie, so the cookie stays
authoritative for the request in flight, which is what §6's answer asked for.
Choosing a language while signed in still records it, so the account learns the
preference without ever fighting the device for it.

### SquadLab: there was no first backend

Favourites were a storage swap. SquadLab was not: it stored **nothing**. The
draft lived in React state and the component warned you on the way out that
leaving would lose the team. Signed out that is still exactly what happens;
signed in, the draft autosaves (debounced — picking an eleven is eighteen
changes in a row) and the tournament it becomes is a durable row. The
`beforeunload` warning is now conditional, which is the honest version of it.

A finished run is **final**. Its result cannot be recreated by playing again
once ratings and pools have moved under it, so the row is never overwritten
after it lands, and it carries the fingerprint it was played against.

That fingerprint is built on the server from `/health`'s `matches` and
`dataThrough`, so **no change to `api/` was needed** — it stays stateless and
anonymous, and nothing in B2 can reach a served probability. The browser does
not send it: the deployment plan does not expose the API publicly, and a
fingerprint the client could choose would be worth nothing anyway.

The saved-runs panel appears on the SquadLab setup screen for somebody signed
in, and **not at all** otherwise — not as a locked panel advertising what an
account would buy. A greyed-out teaser is a gate wearing a disguise, and §1
says the account unlocks saving, not seeing.

### Answered, of the open questions

| Question | What B2 did |
|---|---|
| One run at a time, or many? | Many, with the in-progress one offered as "continue". |
| Can a logged-out visitor still play SquadLab? | Yes, exactly as before, and nothing is saved. |
| Do settings follow the account? | Theme always; language only where the device has no opinion of its own. |
| Usernames? | No. The display name, as decided. |

### Still open after B2

- **Nothing writes `user_progress`, `achievement` or `user_achievement`.** By
  design. The game rules are their own task; the tables are there so it needs
  no migration.
- **`odds_format` is a column nothing reads.** It is in §4's shape and cost one
  nullable column rather than a later migration.
- **An in-progress run is never cleaned up.** Somebody who starts twenty drafts
  and finishes none has twenty rows. Harmless now; a sweep belongs with
  whatever first cares about the count.
- **A finished `result` is a whole tournament.** The list endpoint never sends
  it, but the rows themselves will grow. Worth measuring before it is a
  problem, not after.


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
