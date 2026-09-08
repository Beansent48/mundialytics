import { ImageResponse } from "next/og";

import { api } from "@/lib/api";

export const alt = "Mundialytics match preview";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/**
 * The card people see when a match link is pasted somewhere.
 *
 * Generated per fixture at request time and cached, so every one of the
 * thousands of match URLs shares as a product card rather than a bare link —
 * the cheapest reach this project has, and it costs one file.
 *
 * Deliberately plain CSS: `next/og` renders a subset of flexbox with Satori, so
 * grid, gaps in some positions and CSS variables are not available here. The
 * colours are literals for that reason, and they must be kept in step with the
 * dark theme in globals.css.
 */
export default async function Image({
  params,
}: {
  // Like every other route file in this version, `params` arrives as a promise.
  // Reading it as a plain object silently yields undefined, the API call 404s,
  // and every share falls back to the bare wordmark — which is exactly what it
  // did until this was awaited.
  params: Promise<{ competition: string; fixture: string }>;
}) {
  const { competition, fixture } = await params;

  let match;
  try {
    match = await api.match(competition, fixture);
  } catch {
    match = null;
  }

  const BG = "#08090c";
  const TEXT = "#f2f4f8";
  const DIM = "#6b7385";
  const BRAND = "#8b5cff";

  if (!match) {
    return new ImageResponse(
      (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: BG,
            color: TEXT,
            fontSize: 64,
            letterSpacing: "-0.03em",
          }}
        >
          Mundialytics
        </div>
      ),
      size,
    );
  }

  const p = match.prediction.probabilities;
  const played = match.played && match.score;
  const bars = [
    { w: p.home, color: "#4d8dff" },
    { w: p.draw, color: "#7b8394" },
    { w: p.away, color: "#ff5d6c" },
  ];

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: BG,
          color: TEXT,
          padding: "62px 68px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 12,
              background: BRAND,
              display: "flex",
            }}
          />
          <div
            style={{
              display: "flex",
              fontSize: 22,
              letterSpacing: "0.18em",
              color: DIM,
              textTransform: "uppercase",
            }}
          >
            {`Mundialytics · ${match.competitionName}`}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: 62,
              letterSpacing: "-0.03em",
            }}
          >
            <div style={{ display: "flex" }}>{match.home}</div>
            <div style={{ display: "flex", color: played ? TEXT : DIM, fontSize: played ? 62 : 34 }}>
              {played ? `${match.score!.home} – ${match.score!.away}` : "vs"}
            </div>
            <div style={{ display: "flex" }}>{match.away}</div>
          </div>

          {!played ? (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <div
                style={{
                  display: "flex",
                  height: 74,
                  marginTop: 44,
                  borderRadius: 14,
                  overflow: "hidden",
                }}
              >
                {bars.map((b, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: `${b.w * 100}%`,
                      background: b.color,
                      color: "#ffffff",
                      fontSize: 34,
                    }}
                  >
                    {`${Math.round(b.w * 100)}%`}
                  </div>
                ))}
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginTop: 22,
                  fontSize: 26,
                  color: DIM,
                }}
              >
                <div style={{ display: "flex" }}>
                  {`xG ${match.prediction.expectedGoals.home.toFixed(2)} – ${match.prediction.expectedGoals.away.toFixed(2)}`}
                </div>
                <div style={{ display: "flex" }}>
                  {`Over 2.5 ${Math.round(match.prediction.goals.over25 * 100)}%`}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", fontSize: 24, color: DIM }}>
          Probabilities from a model fitted on 45,938 matches
        </div>
      </div>
    ),
    size,
  );
}
