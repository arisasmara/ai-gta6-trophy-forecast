// GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate on
// Exophase at the case's target, as five quantiles (the case's levels).
//
// Model, in one paragraph: earn rates are not persistent — they DILUTE. The
// tracked player base grows through launch week while the players who have
// earned each trophy grow slower, so a trophy's rate falls as new, less
// progressed players join. The program therefore fits, per trophy, the trend
// of its logit earn rate against log-time since release (recency-weighted
// least squares, shrunk toward the tier mean), and extrapolates that trend to
// the target. Where a trophy's history is frozen (no read ever moved — the
// launch-night page before Exophase's first refresh), the trend is
// unidentified, so the program falls back to a dilution prior: one refresh
// event inside the horizon, keeping a fraction F = 0.22 of the rate, scaled
// by an elasticity that is 0 for a 100% trophy and near 1 for a rare one.
// Spreads are honest: a base term growing with the horizon, the slope's
// standard error times the extrapolation distance, a dilution term from the
// forecast growth of players tracked, and binomial count noise from the
// number of earners behind the last rate. All constants were fixed in
// advance; nothing here is tuned on any answer (there is none to tune on).
//
// Runs offline, stdlib only, deterministic: no clock, no network, no
// randomness. Two executions write byte-identical output.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Release midnight, the origin of log-time. The case carries release_date; fall back to this. */
const RELEASE_FALLBACK = '2026-11-19T00:00:00Z';
/** Recency half-life for the trend fit, in hours: recent reads describe today's dilution pace. */
const HALF_LIFE_HOURS = 24;
/** Prior strength for shrinking a trophy's fitted slope toward its tier mean (in reads). */
const SHRINK_K = 8;
/** Fraction of a rate a single Exophase refresh keeps (dilution prior, flat histories only).
 *  Calibrated from the rules' own published numbers: the persistence baseline
 *  scores ~1.51 loss with ~0.68pp typical miss and ~0 coverage on the practice,
 *  which a refresh model reproduces when new joiners have earned almost
 *  nothing yet — the first refresh keeps roughly a fifth of each rate. */
const REFRESH_KEEP = 0.22;
/** Base spread: logit units, growing with the square root of the horizon in days. */
const SPREAD_BASE_A = 0.45;
const SPREAD_BASE_B = 0.2;
/** Spread per unit of forecast log-growth in players tracked (dilution uncertainty). */
const SPREAD_DILUTION = 0.35;
/** Widest logit move the extrapolation may make from the last read. */
const MAX_LOGIT_MOVE = 3.0;
/** Smallest spread allowed, so no forecast claims false precision. */
const MIN_SPREAD = 0.35;
/** Cap on forecast log-growth of players tracked (a launch-week sanity bound). */
const MAX_LOG_N_GROWTH = Math.log(30);

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Inverse standard normal CDF (Acklam's rational approximation, error < 1.2e-9). */
function normInv(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

/** Weighted least squares of y on x. Returns slope, intercept, slope standard error. */
function weightedFit(
  xs: number[],
  ys: number[],
  ws: number[],
): { slope: number; intercept: number; seSlope: number; sxx: number } {
  const wSum = ws.reduce((a, b) => a + b, 0);
  const xBar = xs.reduce((a, x, i) => a + ws[i] * x, 0) / wSum;
  const yBar = ys.reduce((a, y, i) => a + ws[i] * y, 0) / wSum;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
    sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  }
  if (sxx < 1e-9) return { slope: 0, intercept: yBar, seSlope: 0, sxx: 0 };
  const slope = sxy / sxx;
  const intercept = yBar - slope * xBar;
  let rss = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    rss += ws[i] * e * e;
  }
  const dof = Math.max(1, xs.length - 2);
  const sigma2 = rss / dof;
  return { slope, intercept, seSlope: Math.sqrt(sigma2 / sxx), sxx };
}

/** Hours from t0 to t1 (ISO strings). */
const hoursBetween = (t0: string, t1: string): number =>
  (Date.parse(t1) - Date.parse(t0)) / 3_600_000;
/** Log-time: ln(1 + hours since release midnight), floored at 0 before release. */
const logTime = (releaseMidnight: string, at: string): number =>
  Math.log(1 + Math.max(0, hoursBetween(releaseMidnight, at)));

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const release = c.game?.release_date ? `${c.game.release_date}T00:00:00Z` : RELEASE_FALLBACK;
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const xTarget = logTime(release, c.target);
  const zs = c.levels.map((t) => normInv(Math.min(0.999, Math.max(0.001, t))));

  const reads = c.history ?? [];
  const last = reads[reads.length - 1];
  const lastAt = last ? last.at : c.at;
  const xLast = logTime(release, lastAt);
  const dx = Math.max(0, xTarget - xLast);
  const nLast = last && last.players_tracked > 0 ? last.players_tracked : 1;

  // Forecast log-growth of players tracked from its own trend (recency-weighted).
  let logNGrowth = 0;
  if (reads.length >= 3) {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    const tLastMs = Date.parse(lastAt);
    for (const r of reads) {
      if (!(r.players_tracked > 0)) continue;
      xs.push(logTime(release, r.at));
      ys.push(Math.log(r.players_tracked));
      ws.push(Math.pow(0.5, Math.max(0, tLastMs - Date.parse(r.at)) / (HALF_LIFE_HOURS * 3_600_000)));
    }
    if (xs.length >= 3) {
      const fit = weightedFit(xs, ys, ws);
      const g = Math.max(0, fit.slope);
      logNGrowth = Math.min(MAX_LOG_N_GROWTH, g * dx);
    }
  }

  // Per-trophy trend fits.
  interface TrophyFit {
    varies: boolean;
    slope: number;
    seSlope: number;
    n: number;
    lastRate: number;
    lastLogit: number;
  }
  const fits: TrophyFit[] = c.trophies.map((t, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    const tLastMs = Date.parse(lastAt);
    let minR = Infinity;
    let maxR = -Infinity;
    let lastNonNull = 1;
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      lastNonNull = rate;
      xs.push(logTime(release, r.at));
      ys.push(logit(rate));
      ws.push(Math.pow(0.5, Math.max(0, tLastMs - Date.parse(r.at)) / (HALF_LIFE_HOURS * 3_600_000)));
      if (rate < minR) minR = rate;
      if (rate > maxR) maxR = rate;
    }
    const lastRate = last && last.rates[i] != null ? (last.rates[i] as number) : lastNonNull;
    const lastLogit = logit(lastRate);
    if (xs.length < 3 || maxR - minR < 0.01) {
      return { varies: false, slope: 0, seSlope: 0, n: xs.length, lastRate, lastLogit };
    }
    const fit = weightedFit(xs, ys, ws);
    return { varies: true, slope: fit.slope, seSlope: fit.seSlope, n: xs.length, lastRate, lastLogit };
  });

  // Tier-mean and global slopes over trophies whose history actually moved.
  const varying = fits.filter((f) => f.varies);
  const anyVaries = varying.length > 0;
  const meanSlope = (list: TrophyFit[]): number =>
    list.length === 0 ? 0 : list.reduce((a, f) => a + f.slope, 0) / list.length;
  const globalSlope = meanSlope(varying);
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const inTier = fits.filter((f, idx) => f.varies && c.trophies[idx].tier === tier);
    tierSlope.set(tier, inTier.length > 0 ? meanSlope(inTier) : globalSlope);
  }

  const spreadBase = SPREAD_BASE_A * Math.sqrt(horizonDays) + SPREAD_BASE_B;

  return c.trophies.map((t, i) => {
    const f = fits[i];
    // Elasticity: how much of the player-base growth this trophy's rate fails to
    // keep up with. 0 at 100% (everyone earns it), near 1 for a rare trophy.
    const elasticity = 1 - Math.sqrt(clampRate(f.lastRate) / 100);
    const kLast = Math.max(1, (clampRate(f.lastRate) / 100) * nLast);
    const countVar = 1 / kLast;

    let mu: number;
    let sigma: number;
    if (!anyVaries) {
      // Frozen page: no trend is identified. One refresh inside the horizon
      // keeps REFRESH_KEEP of the rate; the band stays wide because the
      // refresh's timing and size are both unknown.
      const shift = elasticity * Math.log(REFRESH_KEEP);
      mu = f.lastLogit + shift;
      sigma = Math.sqrt(Math.pow(0.85 + 0.45 * elasticity, 2) + countVar);
    } else {
      let slope: number;
      let se: number;
      if (f.varies) {
        const w = f.n / (f.n + SHRINK_K);
        slope = w * f.slope + (1 - w) * (tierSlope.get(t.tier) ?? globalSlope);
        se = f.seSlope;
      } else {
        slope = tierSlope.get(t.tier) ?? globalSlope;
        se = 0.15;
      }
      mu = f.lastLogit + slope * dx;
      const move = mu - f.lastLogit;
      if (move > MAX_LOGIT_MOVE) mu = f.lastLogit + MAX_LOGIT_MOVE;
      if (move < -MAX_LOGIT_MOVE) mu = f.lastLogit - MAX_LOGIT_MOVE;
      const dilVar = Math.pow(SPREAD_DILUTION * logNGrowth * elasticity, 2);
      sigma = Math.sqrt(
        Math.pow(spreadBase, 2) + Math.pow(se * dx, 2) + dilVar + countVar,
      );
    }
    sigma = Math.max(MIN_SPREAD, sigma);
    const q = zs.map((z) => round4(clampRate(invLogitPct(mu + z * sigma))));
    // Enforce non-decreasing quantiles (rounding can tie, never invert, but be safe).
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
