"use client";

import {
  ChevronRight,
  Dices,
  Loader2,
  Play,
  RotateCcw,
  Share2,
  Trophy,
  X,
} from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useCallback, useEffect, useMemo, useState } from "react";

import { LiveMatch } from "@/components/squadlab/live-match";
import { EmptyCard, PlayerCard } from "@/components/squadlab/player-card";
import { ShareCard } from "@/components/squadlab/share-card";
import { TeamSheet } from "@/components/squadlab/team-sheet";
import {
  api,
  type Competition,
  type SquadPlayer,
  type SquadPool,
  type SquadSeason,
} from "@/lib/api";
import { cn, ordinal } from "@/lib/utils";

type Mode = "draft" | "sandbox";
type Phase = "setup" | "building" | "season";
type Slot = { position: string; index: number; key: string };

/** Attack at the top, keeper at the bottom — the way a formation is written. */
const PITCH_ROWS = ["Forward", "Midfielder", "Defender", "Goalkeeper"] as const;

function buildSlots(slots: Record<string, number>): Slot[] {
  const out: Slot[] = [];
  for (const position of PITCH_ROWS) {
    for (let i = 0; i < (slots[position] ?? 0); i++) {
      out.push({ position, index: i, key: `${position}_${i}` });
    }
  }
  return out;
}

/** Stable shuffle: a slot keeps its five until you ask for new ones. */
function sample<T>(items: T[], n: number, seed: number): T[] {
  const arr = [...items];
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, n);
}

export function SquadLab({ competitions }: { competitions: Competition[] }) {
  const t = useTranslations("squadlab");

  const [phase, setPhase] = useState<Phase>("setup");
  const [mode, setMode] = useState<Mode>("draft");
  const [comp, setComp] = useState(competitions[0]?.slug ?? "");

  const [pool, setPool] = useState<SquadPool | null>(null);
  const [poolError, setPoolError] = useState(false);
  const [picks, setPicks] = useState<Record<string, SquadPlayer>>({});
  const [openSlot, setOpenSlot] = useState<string | null>(null);
  const [rerolls, setRerolls] = useState<Record<string, number>>({});
  const [query, setQuery] = useState("");

  const [season, setSeason] = useState<SquadSeason | null>(null);
  const [playing, setPlaying] = useState(false);
  const [seasonError, setSeasonError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(0);

  // Nothing is stored, so leaving loses the team. Warn only once there is
  // something to lose — a prompt over an empty pitch is noise.
  const dirty = Object.keys(picks).length > 0 || season !== null;
  useEffect(() => {
    if (!dirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  async function start() {
    setPhase("building");
    setPool(null);
    setPoolError(false);
    setPicks({});
    setRerolls({});
    setSeason(null);
    try {
      setPool(await api.squadPool(comp));
      // the first slot opens itself: an empty pitch with no prompt is a puzzle
      setOpenSlot("Forward_0");
    } catch {
      setPoolError(true);
    }
  }

  const slots = useMemo(() => (pool ? buildSlots(pool.slots) : []), [pool]);
  const taken = useMemo(
    () => new Set(Object.values(picks).map((p) => p.player)),
    [picks],
  );
  const complete = slots.length > 0 && Object.keys(picks).length === slots.length;

  const candidates = useCallback(
    (slot: Slot): SquadPlayer[] => {
      if (!pool) return [];
      const all = (pool.players[slot.position] ?? []).filter(
        (p) => !taken.has(p.player) || picks[slot.key]?.player === p.player,
      );
      if (mode === "sandbox") {
        const q = query.trim().toLowerCase();
        return (q ? all.filter((p) => p.display.toLowerCase().includes(q)) : all).slice(
          0,
          18,
        );
      }
      const seed =
        slot.key.split("").reduce((a, c) => a + c.charCodeAt(0), 0) * 7919 +
        (rerolls[slot.key] ?? 0) * 104729;
      return sample(all.slice(0, 18), 5, seed).sort((a, b) => b.overall - a.overall);
    },
    [pool, taken, picks, mode, query, rerolls],
  );

  function pick(slotKey: string, player: SquadPlayer) {
    setPicks((prev) => ({ ...prev, [slotKey]: player }));
    setQuery("");
    // walk to the next empty slot so the draft flows without aiming
    const next = slots.find((s) => s.key !== slotKey && !picks[s.key]);
    setOpenSlot(next && next.key !== slotKey ? next.key : null);
  }

  async function playSeason() {
    if (!complete) return;
    setPlaying(true);
    setSeasonError(null);
    try {
      const squad = slots.map((s) => picks[s.key].player);
      const result = await api.playSeason(comp, squad, mode);
      setSeason(result);
      setRevealed(0);
      setPhase("season");
    } catch {
      setSeasonError(t("seasonFailed"));
    } finally {
      setPlaying(false);
    }
  }

  function reset() {
    setPhase("setup");
    setPicks({});
    setSeason(null);
    setRerolls({});
    setOpenSlot(null);
  }

  if (phase === "season" && season) {
    return (
      <SeasonView
        season={season}
        squad={slots.map((s) => picks[s.key]).filter(Boolean)}
        competitionName={competitions.find((c) => c.slug === comp)?.name ?? ""}
        revealed={revealed}
        setRevealed={setRevealed}
        onReset={reset}
      />
    );
  }

  if (phase === "setup") {
    return (
      <div>
        <div className="grid gap-8 sm:grid-cols-2">
          <div>
            <p className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-dim">
              {t("stepMode")}
            </p>
            <div className="mt-3 flex flex-col gap-2">
              {(["draft", "sandbox"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={cn(
                    "rounded-[13px] border px-5 py-4 text-left transition-colors duration-200",
                    mode === m
                      ? "border-brand bg-brand-ghost"
                      : "border-border bg-surface hover:border-border-strong",
                  )}
                >
                  <span className="text-[0.98rem] font-semibold">{t(`mode.${m}`)}</span>
                  <span className="mt-1 block text-[0.83rem] leading-relaxed text-muted">
                    {t(`mode.${m}Lead`)}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <p className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-dim">
              {t("stepCompetition")}
            </p>
            <div className="mt-3 flex flex-col gap-2">
              {competitions.map((c) => (
                <button
                  key={c.slug}
                  type="button"
                  onClick={() => setComp(c.slug)}
                  className={cn(
                    "rounded-[13px] border px-5 py-3.5 text-left text-[0.92rem] font-medium transition-colors duration-200",
                    comp === c.slug
                      ? "border-brand bg-brand-ghost"
                      : "border-border bg-surface text-muted hover:border-border-strong hover:text-text",
                  )}
                >
                  {c.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={start}
          className="mt-10 inline-flex h-13 items-center gap-2 rounded-[12px] bg-brand px-8 py-3.5 text-[1rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover"
        >
          <Play className="size-4" />
          {t("start")}
        </button>
      </div>
    );
  }

  // Thirty seconds of engine time with the team sheet to read, instead of a
  // spinner over an empty pitch.
  if (playing) {
    return <TeamSheet squad={slots.map((s) => picks[s.key])} />;
  }

  /* ── Building the eleven ─────────────────────────────────────────────── */
  const openSlotObj = slots.find((s) => s.key === openSlot);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={reset}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("changeSetup")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {t("setupSummary", {
            mode: t(`mode.${mode}`),
            competition: competitions.find((c) => c.slug === comp)?.name ?? "",
          })}
        </span>
      </div>

      {poolError ? (
        <p className="mt-8 rounded-[var(--radius-card)] border border-border bg-surface px-5 py-10 text-center text-[0.88rem] text-muted">
          {t("poolFailed")}
        </p>
      ) : !pool ? (
        <div className="mv-pitch mt-8 flex min-h-[26rem] items-center justify-center">
          <span className="flex items-center gap-2 text-[0.88rem] text-white/70">
            <Loader2 className="size-4 animate-spin" />
            {t("loadingPool")}
          </span>
        </div>
      ) : (
        <>
          <div className="mv-pitch mt-8 px-4 py-6 sm:px-8 sm:py-9">
            <div className="relative flex flex-col gap-5">
              {PITCH_ROWS.map((position, rowIndex) => {
                const row = slots.filter((s) => s.position === position);
                if (!row.length) return null;
                return (
                  <div
                    key={position}
                    className="mx-auto flex w-full max-w-[42rem] justify-center gap-2 sm:gap-3"
                  >
                    {row.map((slot, i) => {
                      const p = picks[slot.key];
                      const delay = rowIndex * 90 + i * 70;
                      return (
                        <div key={slot.key} className="w-[4.9rem] sm:w-[7rem]">
                          {p ? (
                            <PlayerCard
                              player={p}
                              delay={delay}
                              selected={openSlot === slot.key}
                              onClick={() =>
                                setOpenSlot(openSlot === slot.key ? null : slot.key)
                              }
                            />
                          ) : (
                            <EmptyCard
                              label={t(`positionsShort.${position}`)}
                              delay={delay}
                              active={openSlot === slot.key}
                              onClick={() =>
                                setOpenSlot(openSlot === slot.key ? null : slot.key)
                              }
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>

          {openSlotObj ? (
            <div className="mt-5 rounded-[var(--radius-card)] border border-brand bg-surface p-5">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[0.82rem] font-semibold">
                  {t(`positions.${openSlotObj.position}`)} #{openSlotObj.index + 1}
                </p>
                <div className="flex items-center gap-2">
                  {mode === "draft" ? (
                    <button
                      type="button"
                      onClick={() =>
                        setRerolls((r) => ({
                          ...r,
                          [openSlotObj.key]: (r[openSlotObj.key] ?? 0) + 1,
                        }))
                      }
                      className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-[0.75rem] text-muted transition-colors hover:border-border-strong hover:text-text"
                    >
                      <Dices className="size-3.5" />
                      {t("reroll")}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => setOpenSlot(null)}
                    aria-label={t("close")}
                    className="flex size-7 items-center justify-center rounded-lg border border-border text-muted transition-colors hover:text-text"
                  >
                    <X className="size-3.5" />
                  </button>
                </div>
              </div>

              {mode === "sandbox" ? (
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={t("search")}
                  className="mt-4 w-full rounded-[10px] border border-border bg-bg px-3.5 py-2.5 text-[0.88rem] outline-none placeholder:text-dim focus:border-brand"
                />
              ) : null}

              <div
                // keyed on the slot and reroll so the cards re-deal every time
                key={`${openSlotObj.key}-${rerolls[openSlotObj.key] ?? 0}-${query}`}
                className="mt-4 grid grid-cols-3 gap-2.5 sm:grid-cols-5 lg:grid-cols-6"
              >
                {candidates(openSlotObj).map((p, i) => (
                  <PlayerCard
                    key={p.player}
                    player={p}
                    delay={i * 70}
                    onClick={() => pick(openSlotObj.key, p)}
                  />
                ))}
                {!candidates(openSlotObj).length ? (
                  <p className="col-span-full text-[0.83rem] text-dim">
                    {t("noCandidates")}
                  </p>
                ) : null}
              </div>
            </div>
          ) : null}

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={!complete || playing}
              onClick={playSeason}
              className="inline-flex h-12 items-center gap-2 rounded-[10px] bg-brand px-6 text-[0.95rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover disabled:pointer-events-none disabled:opacity-40"
            >
              {playing ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  {t("playing")}
                </>
              ) : (
                <>
                  {t("play")}
                  <ChevronRight className="size-4" />
                </>
              )}
            </button>
            <span className="text-[0.8rem] text-dim">
              {t("picked", { n: Object.keys(picks).length, total: slots.length })}
            </span>
          </div>

          {seasonError ? (
            <p className="mt-4 text-[0.85rem] text-negative">{seasonError}</p>
          ) : null}

          <p className="mt-6 text-[0.76rem] leading-relaxed text-dim">
            {t("noSaveWarning")}
          </p>
        </>
      )}
    </div>
  );
}

/* ── The season ──────────────────────────────────────────────────────────── */

function SeasonView({
  season,
  squad,
  competitionName,
  revealed,
  setRevealed,
  onReset,
}: {
  season: SquadSeason;
  squad: SquadPlayer[];
  competitionName: string;
  revealed: number;
  setRevealed: (n: number) => void;
  onReset: () => void;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();
  const [sharing, setSharing] = useState(false);
  // The matchday most recently revealed is *being played*: it must not reach
  // the table or the results list yet, or the scoreboard would be reporting a
  // result the reader can already see two panels down.
  const [running, setRunning] = useState(false);
  const [showMatch, setShowMatch] = useState(false);
  const stopRunning = useCallback(() => setRunning(false), []);

  const total = season.matchdays.length;
  const done = revealed >= total && !running;
  const settled = running ? revealed - 1 : revealed;
  const shown = season.matchdays.slice(0, settled);
  const current = revealed > 0 ? season.matchdays[revealed - 1] : null;
  const currentFixture = current?.fixtures.find((f) => f.isSquad) ?? null;

  // Table through the matchdays revealed so far, so the standings move with the
  // season instead of giving the ending away on the first click.
  const table = useMemo(() => {
    const acc: Record<string, { points: number; gf: number; ga: number; played: number }> =
      {};
    for (const md of shown) {
      for (const f of md.fixtures) {
        for (const [team, gf, ga] of [
          [f.home, f.homeGoals, f.awayGoals],
          [f.away, f.awayGoals, f.homeGoals],
        ] as const) {
          const e = (acc[team] ??= { points: 0, gf: 0, ga: 0, played: 0 });
          e.played += 1;
          e.gf += gf;
          e.ga += ga;
          e.points += gf > ga ? 3 : gf === ga ? 1 : 0;
        }
      }
    }
    return Object.entries(acc)
      .map(([team, e]) => ({ team, ...e, gd: e.gf - e.ga }))
      .sort((a, b) => b.points - a.points || b.gd - a.gd || b.gf - a.gf);
  }, [shown]);

  // The API names the club "Your XI"; only the client knows which language the
  // reader is in, so the label is swapped in wherever that name is shown.
  const squadLabel = t("sheetTitle");
  const label = (team: string) => (team === season.teamName ? squadLabel : team);

  const finish = season.finish;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-10 items-center gap-2 rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <RotateCcw className="size-3.5" />
          {t("newTeam")}
        </button>
        {season.replaced ? (
          <span className="text-[0.8rem] text-dim">
            {t("replaced", { team: season.replaced })}
          </span>
        ) : null}
      </div>

      {done ? (
        <div className="mt-8 overflow-hidden rounded-[18px] border border-border bg-surface">
          <div className="relative px-7 py-9 text-center">
            <div
              aria-hidden
              className="pointer-events-none absolute left-1/2 top-0 size-[22rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand opacity-[0.16] blur-[110px]"
            />
            <Trophy className="relative mx-auto size-7 text-warning" />
            <p className="relative mt-4 text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-dim">
              {t("finalTitle")}
            </p>
            <p className="font-display relative mt-2 text-[2.4rem] leading-none sm:text-[3rem]">
              {ordinal(finish?.rank ?? 0, locale)}
            </p>
            <p className="relative mt-3 text-[0.95rem] text-muted">
              {t("finishLine", {
                points: finish?.points ?? 0,
                gf: finish?.goalsFor ?? 0,
                ga: finish?.goalsAgainst ?? 0,
              })}
            </p>

            {season.odds ? (
              <dl className="relative mx-auto mt-7 grid max-w-lg grid-cols-3 gap-px overflow-hidden rounded-[12px] border border-border bg-border">
                {[
                  { k: t("oddsTitle"), v: season.odds.title },
                  { k: t("oddsTop4"), v: season.odds.top4 },
                  { k: t("oddsRelegation"), v: season.odds.relegation },
                ].map((o) => (
                  <div key={o.k} className="bg-bg-elevated px-3 py-3.5">
                    <dt className="text-[0.58rem] font-semibold uppercase tracking-[0.09em] text-dim">
                      {o.k}
                    </dt>
                    <dd className="mt-1.5 text-[1.05rem] font-semibold">
                      {o.v == null ? "—" : `${Math.round(o.v * 100)}%`}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : null}

            <button
              type="button"
              onClick={() => setSharing((s) => !s)}
              className="relative mt-7 inline-flex h-11 items-center gap-2 rounded-[10px] bg-brand px-5 text-[0.88rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover"
            >
              <Share2 className="size-4" />
              {t("shareSeason")}
            </button>
          </div>
        </div>
      ) : null}

      {done && sharing ? (
        <ShareCard
          squad={squad}
          season={season}
          squadLabel={squadLabel}
          competitionName={competitionName}
          onClose={() => setSharing(false)}
        />
      ) : null}

      {showMatch && current && currentFixture ? (
        <div className="mt-8">
          {/* Keyed on the matchday so the clock restarts rather than resuming
              wherever the previous match left it. */}
          <LiveMatch
            key={current.matchday}
            fixture={currentFixture}
            matchday={current.matchday}
            teamName={season.teamName}
            squadLabel={squadLabel}
            running={running}
            onFinish={stopRunning}
          />
        </div>
      ) : null}

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={done || running}
          onClick={() => {
            setRevealed(revealed + 1);
            setRunning(true);
            setShowMatch(true);
          }}
          className="inline-flex h-11 items-center gap-2 rounded-[10px] bg-brand px-5 text-[0.9rem] font-medium text-brand-contrast transition-colors hover:bg-brand-hover disabled:pointer-events-none disabled:opacity-40"
        >
          {revealed === 0 ? t("kickOff") : t("nextMatchday")}
          <ChevronRight className="size-4" />
        </button>
        <button
          type="button"
          disabled={done}
          onClick={() => {
            // Jumping to the end also clears the scoreboard: leaving a
            // mid-season match sitting between the champion card and the final
            // table is a leftover, not a result.
            setRevealed(total);
            setRunning(false);
            setShowMatch(false);
          }}
          className="inline-flex h-11 items-center rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text disabled:pointer-events-none disabled:opacity-40"
        >
          {t("skipToEnd")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {t("matchdayOf", { n: revealed, total })}
        </span>
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("yourResults")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {shown
              .slice()
              .reverse()
              .slice(0, 10)
              .map((md) => {
                const f = md.fixtures.find((x) => x.isSquad);
                if (!f) return null;
                const home = f.home === season.teamName;
                const us = home ? f.homeGoals : f.awayGoals;
                const them = home ? f.awayGoals : f.homeGoals;
                return (
                  <div
                    key={md.matchday}
                    className="flex items-center gap-3 rounded-[11px] border border-border bg-surface px-4 py-2.5"
                  >
                    <span className="w-8 shrink-0 text-[0.68rem] tabular-nums text-dim">
                      J{md.matchday}
                    </span>
                    <span className="flex-1 truncate text-[0.85rem]">
                      {home ? f.away : f.home}
                      <span className="ml-1.5 text-[0.68rem] text-dim">
                        {home ? t("atHome") : t("away")}
                      </span>
                    </span>
                    <span
                      className={cn(
                        "rounded-[7px] border px-2 py-0.5 text-[0.82rem] font-semibold tabular-nums",
                        us > them
                          ? "border-positive/40 bg-positive/10 text-positive"
                          : us === them
                            ? "border-border bg-bg text-muted"
                            : "border-negative/40 bg-negative/10 text-negative",
                      )}
                    >
                      {us}–{them}
                    </span>
                  </div>
                );
              })}
            {!shown.length ? (
              <p className="rounded-[11px] border border-border bg-surface px-4 py-8 text-center text-[0.85rem] text-dim">
                {t("notStarted")}
              </p>
            ) : null}
          </div>
        </section>

        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {done ? t("finalTable") : t("tableSoFar")}
          </h2>
          <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
            <table className="w-full min-w-[22rem] text-[0.82rem]">
              <thead>
                <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                  <th className="px-3 py-2 text-left font-semibold">#</th>
                  <th className="px-3 py-2 text-left font-semibold">{t("colTeam")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colPlayed")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colFor")}</th>
                  <th className="px-2 py-2 text-right font-semibold">
                    {t("colAgainst")}
                  </th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colDiff")}</th>
                  <th className="px-3 py-2 text-right font-semibold">
                    {t("colPoints")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {(done ? season.standings : table).slice(0, 12).map((r, i) => {
                  const isSquad =
                    "isSquad" in r ? r.isSquad : r.team === season.teamName;
                  // The final table names them goalsFor/goalsAgainst, the
                  // running one gf/ga — same figures, two sources.
                  const gf = "goalsFor" in r ? r.goalsFor : r.gf;
                  const ga = "goalsAgainst" in r ? r.goalsAgainst : r.ga;
                  const gd = gf - ga;
                  return (
                    <tr
                      key={r.team}
                      className={cn(
                        "border-b border-border last:border-0",
                        isSquad ? "bg-brand-ghost" : "bg-surface",
                      )}
                    >
                      <td className="px-3 py-2 tabular-nums text-dim">{i + 1}</td>
                      <td
                        className={cn(
                          "px-3 py-2",
                          isSquad ? "font-semibold text-text" : "",
                        )}
                      >
                        {label(r.team)}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {r.played}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {gf}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted">
                        {ga}
                      </td>
                      <td
                        className={cn(
                          "px-2 py-2 text-right tabular-nums",
                          gd > 0 ? "text-positive" : gd < 0 ? "text-negative" : "text-dim",
                        )}
                      >
                        {gd > 0 ? `+${gd}` : gd}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold tabular-nums">
                        {r.points}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {done && season.scorers.length ? (
        <section className="mt-10">
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("topScorers")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {season.scorers.slice(0, 8).map((s, i) => (
              <div
                key={`${s.player}-${i}`}
                className={cn(
                  "flex items-center gap-3 rounded-[11px] border px-4 py-2.5",
                  s.isSquad
                    ? "border-brand/40 bg-brand-ghost"
                    : "border-border bg-surface",
                )}
              >
                <span className="w-5 text-[0.72rem] tabular-nums text-dim">{i + 1}</span>
                <span className="flex-1 truncate text-[0.86rem] font-medium">
                  {s.player}
                </span>
                <span className="truncate text-[0.72rem] text-dim">{s.team}</span>
                <span className="w-8 text-right text-[0.88rem] font-semibold tabular-nums">
                  {s.goals}
                </span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
