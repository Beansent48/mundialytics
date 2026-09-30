"use client";

import {
  ArrowLeftRight,
  ChevronLeft,
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

import { HistoricChampionsView } from "@/components/squadlab/historic-champions";
import { LiveMatch } from "@/components/squadlab/live-match";
import { EmptyCard, PlayerCard } from "@/components/squadlab/player-card";
import { ShareCard } from "@/components/squadlab/share-card";
import { TeamSheet } from "@/components/squadlab/team-sheet";
import {
  api,
  type BracketTie,
  type SquadPlayer,
  type SquadPool,
  type SquadSeason,
} from "@/lib/api";
import { cn, ordinal } from "@/lib/utils";

type Mode = "draft" | "sandbox" | "historic";
type Phase = "setup" | "building" | "lineup" | "season" | "historic";
/** `bench` slots are the seven reserves; `dealIndex` seeds the server's deal
 *  (0-10 the eleven, 11-17 the bench) so no two slots are dealt alike. */
type Slot = { position: string; index: number; key: string; bench: boolean; dealIndex: number };
type Lineup = { xi: SquadPlayer[]; bench: SquadPlayer[] };

/** Attack at the top, keeper at the bottom — the way a formation is written. */
const PITCH_ROWS = ["Forward", "Midfielder", "Defender", "Goalkeeper"] as const;
// Knockout rounds in the order a squad plays them.
const ROUND_ORDER = ["playoff", "r16", "qf", "sf", "final"] as const;

function buildSlots(
  slots: Record<string, number>,
  benchSlots: Record<string, number> = {},
): Slot[] {
  const out: Slot[] = [];
  let n = 0;
  for (const position of PITCH_ROWS) {
    for (let i = 0; i < (slots[position] ?? 0); i++) {
      out.push({ position, index: i, key: `${position}_${i}`, bench: false, dealIndex: n });
      n += 1;
    }
  }
  // the bench, keeper first, as a matchday sheet lists it
  let b = 0;
  for (const position of [...PITCH_ROWS].reverse()) {
    for (let i = 0; i < (benchSlots[position] ?? 0); i++) {
      out.push({ position, index: i, key: `B_${position}_${i}`, bench: true, dealIndex: 11 + b });
      b += 1;
    }
  }
  return out;
}

/** "4-3-3" for an eleven, keeper left out, the way a formation is written. */
function formationOf(xi: SquadPlayer[]): string {
  const n = (p: string) => xi.filter((x) => x.position === p).length;
  return `${n("Defender")}-${n("Midfielder")}-${n("Forward")}`;
}

/** A fresh draft every time. The cards themselves are dealt by the server,
 *  which rolls card kind and then tier, so the rarities on the catalogue
 *  finally decide what you are offered. This seed is the only thing that makes
 *  one draft differ from another, and it never leaves the session. */
function mintSeed(): number {
  const [n] = crypto.getRandomValues(new Uint32Array(1));
  return n;
}

export function SquadLab() {
  const t = useTranslations("squadlab");

  const [phase, setPhase] = useState<Phase>("setup");
  const [mode, setMode] = useState<Mode>("draft");

  const [pool, setPool] = useState<SquadPool | null>(null);
  const [poolError, setPoolError] = useState(false);
  const [picks, setPicks] = useState<Record<string, SquadPlayer>>({});
  const [openSlot, setOpenSlot] = useState<string | null>(null);
  const [rerolls, setRerolls] = useState<Record<string, number>>({});
  const [query, setQuery] = useState("");
  // The draft's own seed, and the cards already dealt, keyed by slot+reroll so
  // a re-render never re-deals a hand the reader is still looking at.
  const [seed, setSeed] = useState(0);
  const [dealt, setDealt] = useState<Record<string, SquadPlayer[]>>({});

  const [lineup, setLineup] = useState<Lineup | null>(null);
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
    if (mode === "historic") {
      setPhase("historic");
      return;
    }
    setPhase("building");
    setPool(null);
    setPoolError(false);
    setPicks({});
    setRerolls({});
    setDealt({});
    setSeed(mintSeed());
    setSeason(null);
    try {
      setPool(await api.squadPool());
      // the first slot opens itself: an empty pitch with no prompt is a puzzle
      setOpenSlot("Forward_0");
    } catch {
      setPoolError(true);
    }
  }

  const slots = useMemo(
    () => (pool ? buildSlots(pool.slots, pool.benchSlots) : []),
    [pool],
  );
  const taken = useMemo(
    () => new Set(Object.values(picks).map((p) => p.player)),
    [picks],
  );
  const complete = slots.length > 0 && Object.keys(picks).length === slots.length;

  const candidates = useCallback(
    (slot: Slot): SquadPlayer[] => {
      if (!pool) return [];
      const free = (p: SquadPlayer) =>
        !taken.has(p.player) || picks[slot.key]?.player === p.player;
      if (mode === "sandbox") {
        const all = (pool.players[slot.position] ?? []).filter(free);
        const q = query.trim().toLowerCase();
        return (q ? all.filter((p) => p.display.toLowerCase().includes(q)) : all).slice(
          0,
          18,
        );
      }
      // Dealt by the server. The filter is a safety net for a hand dealt before
      // the man in it was drafted somewhere else.
      return (dealt[`${slot.key}|${rerolls[slot.key] ?? 0}`] ?? []).filter(free);
    },
    [pool, taken, picks, mode, query, rerolls, dealt],
  );

  // Ask for a hand when a slot opens, or when it is rerolled. Keyed by
  // slot+reroll, so it is fetched once and survives every re-render after.
  const dealKey = openSlot ? `${openSlot}|${rerolls[openSlot] ?? 0}` : null;
  useEffect(() => {
    if (mode !== "draft" || !pool || !openSlot || !dealKey || !seed) return;
    if (dealt[dealKey]) return;
    const slot = slots.find((x) => x.key === openSlot);
    if (!slot) return;

    let cancelled = false;
    api
      .dealSlot(seed, slot.position, slot.dealIndex, rerolls[openSlot] ?? 0,
                Object.values(picks).map((p) => p.player))
      .then((d) => {
        if (!cancelled) setDealt((prev) => ({ ...prev, [dealKey]: d.candidates }));
      })
      .catch(() => {
        // Record the failure as an empty hand. Leaving the key unset would
        // read as "still dealing" and spin for ever.
        if (!cancelled) setDealt((prev) => ({ ...prev, [dealKey]: [] }));
      });
    return () => {
      cancelled = true;
    };
    // `picks` is read, not watched: a hand already dealt must not be re-dealt
    // under the reader because another slot was filled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, pool, openSlot, dealKey, seed, slots, dealt]);

  function pick(slotKey: string, player: SquadPlayer) {
    setPicks((prev) => ({ ...prev, [slotKey]: player }));
    setQuery("");
    // walk to the next empty slot so the draft flows without aiming
    const next = slots.find((s) => s.key !== slotKey && !picks[s.key]);
    setOpenSlot(next && next.key !== slotKey ? next.key : null);
  }

  /** From a finished draft to the team sheet, where the eleven can be changed. */
  function toLineup() {
    if (!complete) return;
    setLineup({
      xi: slots.filter((s) => !s.bench).map((s) => picks[s.key]),
      bench: slots.filter((s) => s.bench).map((s) => picks[s.key]),
    });
    setPhase("lineup");
  }

  async function playChampions() {
    if (!lineup) return;
    setPlaying(true);
    setSeasonError(null);
    try {
      const result = await api.playChampions(
        lineup.xi.map((p) => p.player),
        mode,
        lineup.bench.map((p) => p.player),
      );
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
    setLineup(null);
    setSeason(null);
    setRerolls({});
    setDealt({});
    setOpenSlot(null);
  }

  if (phase === "historic") {
    return <HistoricChampionsView onReset={reset} />;
  }

  if (phase === "season" && season) {
    return (
      <SeasonView
        season={season}
        squad={lineup?.xi ?? []}
        revealed={revealed}
        setRevealed={setRevealed}
        onReset={reset}
      />
    );
  }

  if (phase === "setup") {
    return (
      <div>
        <div className="max-w-xl">
          <p className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-dim">
            {t("stepMode")}
          </p>
          <div className="mt-3 flex flex-col gap-2">
            {(["draft", "sandbox", "historic"] as const).map((m) => (
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

          <p className="mt-6 flex items-start gap-2 rounded-[12px] border border-border bg-surface px-4 py-3 text-[0.83rem] leading-relaxed text-muted">
            <Trophy className="mt-0.5 size-4 shrink-0 text-warning" />
            {t("championsNote")}
          </p>
        </div>

        <button
          type="button"
          onClick={start}
          className="mt-8 inline-flex h-13 items-center gap-2 rounded-[12px] bg-brand px-8 py-3.5 text-[1rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover"
        >
          <Play className="size-4" />
          {mode === "historic" ? t("historic.start") : t("start")}
        </button>
      </div>
    );
  }

  // Engine time with the team sheet to read, instead of a spinner over an
  // empty pitch.
  if (playing && lineup) {
    return <TeamSheet squad={lineup.xi} />;
  }

  if (phase === "lineup" && lineup && pool) {
    return (
      <LineupEditor
        lineup={lineup}
        setLineup={setLineup}
        formations={pool.formations ?? ["4-3-3"]}
        roleWeights={pool.roleWeights}
        substatLabels={pool.substatLabels}
        onBack={() => setPhase("building")}
        onPlay={playChampions}
        error={seasonError}
      />
    );
  }

  /* ── Building the eleven ─────────────────────────────────────────────── */
  const openSlotObj = slots.find((s) => s.key === openSlot);
  // The cap is served, not hardcoded: unlimited rerolls make rarity pointless,
  // because you just spin until the icon turns up.
  const rerollsLeft = openSlotObj
    ? (pool?.maxRerolls ?? 3) - (rerolls[openSlotObj.key] ?? 0)
    : 0;
  // Derived, not stored: a hand is "on its way" exactly while its key has no
  // entry yet, and an effect that sets state on its own way in trips the
  // set-state-in-effect rule.
  const dealing = mode === "draft" && !!dealKey && dealt[dealKey] === undefined;

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
          {t("setupSummary", { mode: t(`mode.${mode}`) })}
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
                const row = slots.filter((s) => s.position === position && !s.bench);
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
                              roleWeights={pool.roleWeights}
                              substatLabels={pool.substatLabels}
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

          <div className="mt-4 rounded-[var(--radius-card)] border border-border bg-surface px-4 py-4 sm:px-6">
            <p className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-muted">
              {t("bench")}
            </p>
            <p className="mt-1 text-[0.78rem] text-dim">{t("benchLead")}</p>
            <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-7 sm:gap-3">
              {slots
                .filter((s) => s.bench)
                .map((slot, i) => {
                  const p = picks[slot.key];
                  return (
                    <div key={slot.key}>
                      {p ? (
                        <PlayerCard
                          player={p}
                          delay={i * 50}
                          selected={openSlot === slot.key}
                          roleWeights={pool.roleWeights}
                          substatLabels={pool.substatLabels}
                          onClick={() =>
                            setOpenSlot(openSlot === slot.key ? null : slot.key)
                          }
                        />
                      ) : (
                        <EmptyCard
                          label={t(`positionsShort.${slot.position}`)}
                          delay={i * 50}
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
          </div>

          {openSlotObj ? (
            <div className="mt-5 rounded-[var(--radius-card)] border border-brand bg-surface p-5">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[0.82rem] font-semibold">
                  {openSlotObj.bench ? `${t("bench")} · ` : ""}
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
                      disabled={rerollsLeft <= 0 || dealing}
                      className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-[0.75rem] text-muted transition-colors hover:border-border-strong hover:text-text disabled:pointer-events-none disabled:opacity-40"
                    >
                      <Dices className="size-3.5" />
                      {t("rerollLeft", { n: rerollsLeft })}
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
                    roleWeights={pool.roleWeights}
                    substatLabels={pool.substatLabels}
                    onClick={() => pick(openSlotObj.key, p)}
                  />
                ))}
                {!candidates(openSlotObj).length ? (
                  <p className="col-span-full text-[0.83rem] text-dim">
                    {dealing ? t("dealing") : t("noCandidates")}
                  </p>
                ) : null}
              </div>
            </div>
          ) : null}

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={!complete || playing}
              onClick={toLineup}
              className="inline-flex h-12 items-center gap-2 rounded-[10px] bg-brand px-6 text-[0.95rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover disabled:pointer-events-none disabled:opacity-40"
            >
              {playing ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  {t("playing")}
                </>
              ) : (
                <>
                  {t("toLineup")}
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

/* ── The team sheet before kick-off ────────────────────────────────────────── */

/**
 * Your eleven and your bench, swappable. Tap a starter and a reserve (either
 * order) to swap them. A swap across lines is allowed when it leaves a real
 * formation — a defender for a forward turns 4-3-3 into 5-3-2 — and refused
 * when it would not (two keepers, six at the back).
 */
function LineupEditor({
  lineup,
  setLineup,
  formations,
  roleWeights,
  substatLabels,
  onBack,
  onPlay,
  error,
}: {
  lineup: Lineup;
  setLineup: (l: Lineup) => void;
  formations: string[];
  roleWeights: SquadPool["roleWeights"];
  substatLabels: SquadPool["substatLabels"];
  onBack: () => void;
  onPlay: () => void;
  error: string | null;
}) {
  const t = useTranslations("squadlab");
  const [picked, setPicked] = useState<{ group: "xi" | "bench"; i: number } | null>(null);
  const [refused, setRefused] = useState(false);

  const shape = formationOf(lineup.xi);
  const avg = Math.round(
    lineup.xi.reduce((a, p) => a + p.overall, 0) / Math.max(lineup.xi.length, 1),
  );

  function tap(group: "xi" | "bench", i: number) {
    setRefused(false);
    if (!picked || picked.group === group) {
      setPicked(picked && picked.group === group && picked.i === i ? null : { group, i });
      return;
    }
    const xiIdx = group === "xi" ? i : picked.i;
    const benchIdx = group === "bench" ? i : picked.i;
    const xi = lineup.xi.slice();
    const bench = lineup.bench.slice();
    const out = xi[xiIdx];
    xi[xiIdx] = bench[benchIdx];
    bench[benchIdx] = out;
    const keepers = xi.filter((p) => p.position === "Goalkeeper").length;
    if (keepers !== 1 || !formations.includes(formationOf(xi))) {
      setRefused(true);
      setPicked(null);
      return;
    }
    setLineup({ xi, bench });
    setPicked(null);
  }

  const card = (p: SquadPlayer, group: "xi" | "bench", i: number, delay: number) => (
    <PlayerCard
      player={p}
      delay={delay}
      selected={picked?.group === group && picked.i === i}
      roleWeights={roleWeights}
      substatLabels={substatLabels}
      onClick={() => tap(group, i)}
    />
  );

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-9 items-center gap-2 rounded-[10px] border border-border px-3.5 text-[0.83rem] text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          <ChevronLeft className="size-3.5" />
          {t("backToDraft")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {t("formation", { f: shape })} · {t("sheetOverall")} {avg}
        </span>
      </div>

      <div className="mt-5 max-w-2xl">
        <p className="font-display text-[1.5rem] leading-tight">{t("lineupTitle")}</p>
        <p className="mt-1.5 flex items-start gap-2 text-[0.85rem] leading-relaxed text-muted">
          <ArrowLeftRight className="mt-0.5 size-4 shrink-0 text-brand" />
          {t("lineupLead")}
        </p>
      </div>

      <div className="mv-pitch mt-6 px-4 py-6 sm:px-8 sm:py-9">
        <div className="relative flex flex-col gap-5">
          {PITCH_ROWS.map((position, rowIndex) => {
            const row = lineup.xi
              .map((p, i) => ({ p, i }))
              .filter(({ p }) => p.position === position);
            if (!row.length) return null;
            return (
              <div
                key={position}
                className="mx-auto flex w-full max-w-[42rem] justify-center gap-2 sm:gap-3"
              >
                {row.map(({ p, i }, k) => (
                  <div key={p.player} className="w-[4.9rem] sm:w-[7rem]">
                    {card(p, "xi", i, rowIndex * 90 + k * 70)}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-4 rounded-[var(--radius-card)] border border-border bg-surface px-4 py-4 sm:px-6">
        <p className="text-[0.62rem] font-semibold uppercase tracking-[0.13em] text-muted">
          {t("bench")}
        </p>
        <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-7 sm:gap-3">
          {lineup.bench.map((p, i) => (
            <div key={p.player}>{card(p, "bench", i, i * 50)}</div>
          ))}
        </div>
      </div>

      {refused ? (
        <p className="mt-3 text-[0.83rem] text-negative">{t("invalidSwap")}</p>
      ) : null}

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onPlay}
          className="inline-flex h-12 items-center gap-2 rounded-[10px] bg-brand px-6 text-[0.95rem] font-medium text-brand-contrast transition-all duration-200 hover:bg-brand-hover"
        >
          {t("play")}
          <ChevronRight className="size-4" />
        </button>
      </div>
      {error ? <p className="mt-4 text-[0.85rem] text-negative">{error}</p> : null}
    </div>
  );
}

/* ── The Champions run ─────────────────────────────────────────────────────── */

function SeasonView({
  season,
  squad,
  revealed,
  setRevealed,
  onReset,
}: {
  season: SquadSeason;
  squad: SquadPlayer[];
  revealed: number;
  setRevealed: (n: number) => void;
  onReset: () => void;
}) {
  const t = useTranslations("squadlab");
  const locale = useLocale();
  const [sharing, setSharing] = useState(false);
  // The match most recently revealed is *being played*: it must not reach the
  // results list yet, or the list would report a result the reader can still
  // see running two panels down.
  const [running, setRunning] = useState(false);
  const [showMatch, setShowMatch] = useState(false);
  const stopRunning = useCallback(() => setRunning(false), []);

  const matches = season.matches;
  const total = matches.length;
  const done = revealed >= total && !running;
  const settled = running ? revealed - 1 : revealed;
  const shown = matches.slice(0, settled);
  const current = revealed > 0 ? matches[revealed - 1] : null;

  // The API names the club "Tu Equipo"; only the client knows the reader's
  // language, so the label is swapped in wherever that name is shown.
  const squadLabel = t("sheetTitle");
  const label = (team: string) => (team === season.teamName ? squadLabel : team);

  // The squad's own knockout ties, in order, plus the final for the champion.
  const ownTies = useMemo(() => {
    const out: BracketTie[] = [];
    for (const r of ROUND_ORDER) {
      for (const tie of season.bracket[r] ?? []) {
        if (tie.isSquad) out.push(tie);
      }
    }
    return out;
  }, [season.bracket]);
  const finalTie = season.bracket.final?.[0] ?? null;

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
            <p className="font-display relative mt-2 text-[2rem] leading-tight sm:text-[2.6rem]">
              {season.stage}
            </p>
            <p className="relative mt-3 text-[0.95rem] text-muted">
              {t("leaguePhaseRankLine", {
                rank: ordinal(season.leaguePhaseRank, locale),
              })}
              {" · "}
              {t("championLine", { team: label(season.champion) })}
            </p>

            <dl className="relative mx-auto mt-7 grid max-w-md grid-cols-2 gap-px overflow-hidden rounded-[12px] border border-border bg-border">
              <div className="bg-bg-elevated px-3 py-3.5">
                <dt className="text-[0.58rem] font-semibold uppercase tracking-[0.09em] text-dim">
                  {t("squadElo")}
                </dt>
                <dd className="mt-1.5 text-[1.05rem] font-semibold tabular-nums">
                  {Math.round(season.squadElo)}
                </dd>
              </div>
              <div className="bg-bg-elevated px-3 py-3.5">
                <dt className="text-[0.58rem] font-semibold uppercase tracking-[0.09em] text-dim">
                  {t("leaguePhasePos")}
                </dt>
                <dd className="mt-1.5 text-[1.05rem] font-semibold tabular-nums">
                  {ordinal(season.leaguePhaseRank, locale)}
                </dd>
              </div>
            </dl>

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
          onClose={() => setSharing(false)}
        />
      ) : null}

      {showMatch && current ? (
        <div className="mt-8">
          {/* Keyed on the match index so the clock restarts rather than resuming
              wherever the previous match left it. */}
          <LiveMatch
            key={revealed}
            fixture={current}
            label={current.stageLabel}
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
          {revealed === 0 ? t("kickOff") : t("nextMatch")}
          <ChevronRight className="size-4" />
        </button>
        <button
          type="button"
          disabled={done}
          onClick={() => {
            // Jumping to the end also clears the scoreboard: a mid-run match
            // left between the trophy card and the tables is a leftover.
            setRevealed(total);
            setRunning(false);
            setShowMatch(false);
          }}
          className="inline-flex h-11 items-center rounded-[10px] border border-border px-4 text-[0.85rem] text-muted transition-colors hover:border-border-strong hover:text-text disabled:pointer-events-none disabled:opacity-40"
        >
          {t("skipToEnd")}
        </button>
        <span className="text-[0.8rem] text-dim">
          {t("matchOf", { n: revealed, total })}
        </span>
      </div>

      <div className="mt-8 flex flex-col gap-10">
        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("leaguePhaseTable")}
          </h2>
          <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
            <table className="w-full min-w-[22rem] text-[0.82rem]">
              <thead>
                <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                  <th className="px-3 py-2 text-left font-semibold">#</th>
                  <th className="px-3 py-2 text-left font-semibold">{t("colTeam")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colPlayed")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colFor")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colAgainst")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colDiff")}</th>
                  <th className="px-3 py-2 text-right font-semibold">{t("colPoints")}</th>
                </tr>
              </thead>
              <tbody>
                {season.leaguePhase.map((r) => (
                  <tr
                    key={r.team}
                    className={cn(
                      "border-b border-border last:border-0",
                      // Where the table actually splits: 1-8 go straight to the
                      // last 16, 9-24 play the knockout play-off, 25-36 are out.
                      // Without the rules drawn on it, 36 rows is just a list.
                      r.rank === 9 || r.rank === 25
                        ? "border-t-2 border-t-border-strong"
                        : "",
                      r.isSquad ? "bg-brand-ghost" : "bg-surface",
                    )}
                  >
                    <td className="px-3 py-2 tabular-nums text-dim">{r.rank}</td>
                    <td className={cn("px-3 py-2", r.isSquad ? "font-semibold text-text" : "")}>
                      {label(r.team)}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">
                      {r.played}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">
                      {r.goalsFor}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">
                      {r.goalsAgainst}
                    </td>
                    <td
                      className={cn(
                        "px-2 py-2 text-right tabular-nums",
                        r.goalDiff > 0
                          ? "text-positive"
                          : r.goalDiff < 0
                            ? "text-negative"
                            : "text-dim",
                      )}
                    >
                      {r.goalDiff > 0 ? `+${r.goalDiff}` : r.goalDiff}
                    </td>
                    <td className="px-3 py-2 text-right font-semibold tabular-nums">
                      {r.points}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[0.7rem] text-dim">{t("tableLegend")}</p>
        </section>
        <section>
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("yourResults")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {shown
              .slice()
              .reverse()
              .map((f, i) => {
                const home = f.home === season.teamName;
                const us = home ? f.homeGoals : f.awayGoals;
                const them = home ? f.awayGoals : f.homeGoals;
                return (
                  <div
                    key={`${f.stageLabel}-${i}`}
                    className="flex items-center gap-3 rounded-[11px] border border-border bg-surface px-4 py-2.5"
                  >
                    <span className="w-24 shrink-0 truncate text-[0.66rem] uppercase tracking-[0.06em] text-dim">
                      {f.stageLabel}
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

          {done && ownTies.length ? (
            <div className="mt-6">
              <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
                {t("yourRun")}
              </h2>
              <div className="mt-3 flex flex-col gap-1.5">
                {ownTies.map((tie, i) => {
                  const opp = tie.teamA === season.teamName ? tie.teamB : tie.teamA;
                  const won = tie.winner === season.teamName;
                  // the API writes the aggregate from team A's side; read it
                  // from yours, or a won final printed as "0-2"
                  const agg =
                    tie.teamA === season.teamName
                      ? tie.agg
                      : tie.agg.split("-").reverse().join("-");
                  return (
                    <div
                      key={`${tie.round}-${i}`}
                      className={cn(
                        "flex items-center gap-3 rounded-[11px] border px-4 py-2.5",
                        won
                          ? "border-positive/40 bg-positive/5"
                          : "border-negative/40 bg-negative/5",
                      )}
                    >
                      <span className="w-20 shrink-0 truncate text-[0.66rem] uppercase tracking-[0.06em] text-dim">
                        {tie.roundLabel}
                      </span>
                      <span className="flex-1 truncate text-[0.85rem]">{label(opp)}</span>
                      <span className="text-[0.7rem] text-dim">
                        {agg}
                        {tie.note ? ` ${tie.note}` : ""}
                      </span>
                      <span
                        className={cn(
                          "rounded-[6px] px-1.5 py-0.5 text-[0.68rem] font-semibold",
                          won ? "bg-positive/15 text-positive" : "bg-negative/12 text-negative",
                        )}
                      >
                        {won ? t("advanced") : t("eliminated")}
                      </span>
                    </div>
                  );
                })}
                {/* Your own final is already in the run above; this block is
                    for the final you did not reach, so the champion is still
                    named. Showing both printed the same tie twice. */}
                {finalTie && !finalTie.isSquad ? (
                  <div className="mt-1 flex items-center gap-3 rounded-[11px] border border-warning/40 bg-warning/5 px-4 py-2.5">
                    <span className="w-20 shrink-0 truncate text-[0.66rem] uppercase tracking-[0.06em] text-warning">
                      {finalTie.roundLabel}
                    </span>
                    <span className="flex-1 truncate text-[0.85rem]">
                      {label(finalTie.teamA)} — {label(finalTie.teamB)}
                    </span>
                    <span className="text-[0.7rem] text-dim">
                      {finalTie.leg1}
                      {finalTie.note ? ` ${finalTie.note}` : ""}
                    </span>
                    <span className="rounded-[6px] bg-warning/15 px-1.5 py-0.5 text-[0.68rem] font-semibold text-warning">
                      {label(finalTie.winner)}
                    </span>
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}
        </section>
      </div>

      {done && season.scorers.length ? (
        <section className="mt-10">
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("topScorers")}
          </h2>
          <div className="mt-3 flex flex-col gap-1.5">
            {season.scorers.slice(0, 10).map((s, i) => (
              <div
                key={`${s.player}-${i}`}
                className={cn(
                  "flex items-center gap-3 rounded-[11px] border px-4 py-2.5",
                  // the highlight is what makes YOUR man stand out; before this
                  // the list was only ever your own eleven, so everything glowed
                  s.isSquad
                    ? "border-brand/40 bg-brand-ghost"
                    : "border-border bg-surface",
                )}
              >
                <span className="w-5 text-[0.72rem] tabular-nums text-dim">{i + 1}</span>
                <span className="flex-1 truncate text-[0.86rem] font-medium">
                  {s.player}
                  <span className="ml-2 text-[0.68rem] font-normal text-dim">
                    {label(s.team)}
                  </span>
                </span>
                {s.assists ? (
                  <span className="text-[0.68rem] text-dim">
                    {t("assistShort", { n: s.assists })}
                  </span>
                ) : null}
                <span className="w-8 text-right text-[0.88rem] font-semibold tabular-nums">
                  {s.goals}
                </span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {done && season.squadTotals?.length ? (
        <section className="mt-10">
          <h2 className="text-[0.66rem] font-semibold uppercase tracking-[0.13em] text-muted">
            {t("squadTotals")}
          </h2>
          <div className="mt-3 overflow-x-auto rounded-[var(--radius-card)] border border-border">
            <table className="w-full min-w-[28rem] text-[0.82rem]">
              <thead>
                <tr className="border-b border-border bg-bg-elevated text-[0.6rem] uppercase tracking-[0.07em] text-dim">
                  <th className="px-3 py-2 text-left font-semibold">{t("colPlayer")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colPlayed")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colMinutes")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colGoals")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colAssists")}</th>
                  <th className="px-2 py-2 text-right font-semibold">{t("colCards")}</th>
                  <th className="px-3 py-2 text-right font-semibold">{t("colRating")}</th>
                </tr>
              </thead>
              <tbody>
                {season.squadTotals.map((r) => (
                  <tr key={r.player} className="border-b border-border bg-surface last:border-0">
                    <td className="px-3 py-2">
                      {r.player}
                      {r.injuries ? (
                        <span className="ml-1.5 text-[0.68rem] text-negative">
                          {t("injuriesShort", { n: r.injuries })}
                        </span>
                      ) : null}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">{r.apps}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">{r.minutes}</td>
                    <td className="px-2 py-2 text-right tabular-nums">{r.goals}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">{r.assists}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted">
                      {r.yellows}
                      {r.reds ? <span className="ml-1 text-negative">+{r.reds}R</span> : null}
                    </td>
                    <td className="px-3 py-2 text-right font-semibold tabular-nums">
                      {r.rating != null ? r.rating.toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
