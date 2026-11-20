// GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01), v2.
//
// Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate on
// Exophase at the case's target, as five quantiles (the case's levels).
//
// Model, in one paragraph: earn rates are not persistent — they DILUTE, and
// they dilute in DISCRETE steps, not smoothly. Exophase refreshes a game's
// counts on its own schedule: the page sits unchanged between refreshes and
// jumps at each one (the house saw 638 players tracked, every rate frozen,
// from 2026-11-18T23:48Z through 04:00Z, then a refresh by 04:30Z: 3,868
// players, every rate but the one at 100% lower — RULES.txt). The answer is
// the rates the page shows at the target, i.e. as of Exophase's last refresh
// before it. So the program works in COUNT space and steps through refreshes:
//
//   * Refresh detection: a refresh is a step where players_tracked jumps by
//     >= 25% between consecutive reads. From the most recent observed refresh
//     the program measures, per tier, the keep factor (rate after / rate
//     before, median over the tier's trophies) and the implied new-joiner
//     earn propensity f (the fraction of the old rate at which a new joiner
//     holds the trophy), plus the within-tier dispersion of log-keep, which
//     prices how uncertain future refreshes are.
//   * Refresh cadence: the median gap between observed refreshes (with one
//     observed, the gap from release to it). Refresh times inside the horizon
//     are predicted by stepping that cadence forward from the last observed
//     refresh; the forecast is evaluated as of the LAST predicted refresh at
//     or before target + 60 min, because the page does not move after it.
//   * Frozen history (no read ever moved — the practice cases): no refresh
//     has been observed, so the program falls back to a dilution prior for
//     exactly one refresh inside the horizon. The refresh size is the
//     published one (3,868 / 638 = 6.0627x players tracked), and each
//     trophy's new earners are p * (new joiners), with p = f_tier * rate and
//     f_tier = 0.04 * {platinum 0.6, gold 0.8, silver 1.0, bronze 1.2}.
//     f rises toward 1 as a rate approaches 100% (a near-universal trophy
//     stays near-universal). The 0.04 scale and the tight frozen spread
//     (0.25 logit) were calibrated on a family of synthetic refresh truths
//     (uniform f in [0, 0.10] plus rate-dependent ones), anchored on v1's
//     published server practice loss of 0.327, which sits at effective
//     f ~= 0.025; across that family this prior scores ~0.05-0.11 where v1
//     scored ~0.27-0.35. The spread is tight because the keep factor is now
//     much better identified than v1's flat 0.22^elasticity guess was.
//   * Varying history: the per-trophy recency-weighted logit trend against
//     log-time since release (shrunk toward the tier mean, as in v1) is the
//     base, but it is evaluated at the last predicted refresh before the
//     target, not at the target — the page lags the true state by up to one
//     refresh interval. The spread adds a refresh term from the observed
//     keep dispersion, growing with the number of future refreshes.
//
// All constants were fixed in advance from the rules and the calibration
// family above; nothing here is tuned on any answer (there is none to tune
// on). Runs offline, stdlib only, deterministic: no clock, no network, no
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
/** Published first-refresh size: players tracked 638 -> 3,868 (RULES.txt).
 *  The frozen-history prior assumes one refresh of this size inside the horizon. */
const REFRESH_N_RATIO = 3868 / 638;
/** A refresh is a step where players_tracked jumps by at least this ratio between reads. */
const REFRESH_JUMP = 1.25;
/** Base new-joiner earn propensity for the frozen-history prior (see header).
 *  Per tier it is F_PRIOR_BASE * TIER_F_MULT[tier]. */
const F_PRIOR_BASE = 0.04;
const TIER_F_MULT: Record<string, number> = {
  platinum: 0.6,
  gold: 0.8,
  silver: 1.0,
  bronze: 1.2,
};
/** Frozen-history spread (logit): the keep factor is now well identified, so the
 *  band is much tighter than v1's 0.85+. Calibrated on the refresh-truth family. */
const SIGMA_FROZEN = 0.25;
/** Base spread: logit units, growing with the square root of the horizon in days. */
const SPREAD_BASE_A = 0.45;
const SPREAD_BASE_B = 0.2;
/** Spread per unit of forecast log-growth in players tracked (dilution uncertainty). */
const SPREAD_DILUTION = 0.35;
/** Cap on the extra refresh variance added to the trend branch's spread. */
const REFRESH_VAR_CAP = 0.09;
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

/** Median of numbers (sorted copy; deterministic). */
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Hours from t0 to t1 (ISO strings). */
const hoursBetween = (t0: string, t1: string): number =>
  (Date.parse(t1) - Date.parse(t0)) / 3_600_000;
/** Log-time: ln(1 + hours since release midnight), floored at 0 before release. */
const logTime = (releaseMidnight: string, at: string): number =>
  Math.log(1 + Math.max(0, hoursBetween(releaseMidnight, at)));

interface Refresh {
  at: string;
  nBefore: number;
  nAfter: number;
  idx: number;
}

/** Detect Exophase refreshes: steps where players_tracked jumps by >= REFRESH_JUMP. */
function detectRefreshes(reads: Read[]): Refresh[] {
  const out: Refresh[] = [];
  for (let j = 1; j < reads.length; j++) {
    const before = reads[j - 1];
    const after = reads[j];
    if (before.players_tracked > 0 && after.players_tracked >= REFRESH_JUMP * before.players_tracked) {
      out.push({ at: after.at, nBefore: before.players_tracked, nAfter: after.players_tracked, idx: j });
    }
  }
  return out;
}

/**
 * Per-tier keep factors and new-joiner propensities from one observed refresh,
 * plus the within-tier dispersion of log-keep (how uncertain future refreshes are).
 * keep_i = rate after / rate before; f_tier solves keep = (N_b + dN * f) / N_a.
 * The 100% trophy is excluded: it stays 100% by construction.
 */
function refreshStats(
  reads: Read[],
  r: Refresh,
  trophies: Trophy[],
): { fTier: Map<string, number>; dispersion: number } {
  const before = reads[r.idx - 1];
  const after = reads[r.idx];
  const dN = r.nAfter - r.nBefore;
  const byTier = new Map<string, number[]>();
  const residSq: number[] = [];
  for (const tier of new Set(trophies.map((t) => t.tier))) {
    const keeps: number[] = [];
    trophies.forEach((t, i) => {
      if (t.tier !== tier) return;
      const rb = before.rates[i];
      const ra = after.rates[i];
      if (rb === null || rb === undefined || ra === null || ra === undefined) return;
      if (rb <= 0 || rb >= 100 || ra <= 0) return;
      keeps.push(ra / rb);
    });
    if (keeps.length > 0) byTier.set(tier, keeps);
  }
  const fTier = new Map<string, number>();
  for (const [tier, keeps] of byTier) {
    const keep = median(keeps);
    const logKeep = Math.log(Math.max(0.01, keep));
    for (const k of keeps) residSq.push(Math.pow(Math.log(Math.max(0.01, k)) - logKeep, 2));
    const f = dN > 0 ? (keep * r.nAfter - r.nBefore) / dN : 0;
    fTier.set(tier, Math.min(1, Math.max(0, f)));
  }
  const dispersion = residSq.length > 0
    ? Math.sqrt(residSq.reduce((a, b) => a + b, 0) / residSq.length)
    : 0;
  return { fTier, dispersion };
}

/**
 * New-joiner earn propensity for a trophy at rate `ratePct`, given the tier's
 * base ratio f0: p = f * rate, where f rises from f0 toward 1 as the rate
 * approaches 100% (a near-universal trophy stays near-universal after a
 * refresh, while a rare one keeps only f0 of its rate).
 */
function earnProb(f0: number, ratePct: number): number {
  const r = clampRate(ratePct) / 100;
  const f = f0 + (1 - f0) * r * r;
  return Math.min(1, f * r);
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const release = c.game?.release_date ? `${c.game.release_date}T00:00:00Z` : RELEASE_FALLBACK;
  const zs = c.levels.map((t) => normInv(Math.min(0.999, Math.max(0.001, t))));

  const reads = c.history ?? [];
  const last = reads[reads.length - 1];
  const lastAt = last ? last.at : c.at;
  const xLast = logTime(release, lastAt);
  const nLast = last && last.players_tracked > 0 ? last.players_tracked : 1;

  // --- Refresh machinery -------------------------------------------------
  const refreshes = detectRefreshes(reads);
  const lastRefresh = refreshes.length > 0 ? refreshes[refreshes.length - 1] : null;
  let fTierObserved: Map<string, number> | null = null;
  let dispersion = 0;
  if (lastRefresh) {
    const st = refreshStats(reads, lastRefresh, c.trophies);
    fTierObserved = st.fTier;
    dispersion = st.dispersion;
  }
  // Refresh cadence: median gap between observed refreshes; with one observed,
  // the gap from release to it. Null when no refresh was ever observed.
  let cadenceMs = 0;
  if (refreshes.length >= 2) {
    const gaps: number[] = [];
    for (let j = 1; j < refreshes.length; j++) {
      gaps.push(Date.parse(refreshes[j].at) - Date.parse(refreshes[j - 1].at));
    }
    cadenceMs = median(gaps);
  } else if (refreshes.length === 1) {
    cadenceMs = Date.parse(refreshes[0].at) - Date.parse(release);
  }
  // Predicted refresh times inside the horizon: step the cadence forward from
  // the last observed refresh, keeping only times after the last read and at
  // or before target + 60 min (the answering read's window).
  const answerWindowEndMs = Date.parse(c.target) + 3_600_000;
  const predicted: number[] = [];
  if (cadenceMs > 0 && lastRefresh) {
    let t = Date.parse(lastRefresh.at) + cadenceMs;
    let guard = 0;
    while (t <= answerWindowEndMs && guard < 100) {
      if (t > Date.parse(lastAt)) predicted.push(t);
      t += cadenceMs;
      guard++;
    }
  }
  const nFuture = predicted.length;
  // The forecast is evaluated as of the last predicted refresh before the
  // answering window ends; with none predicted, as of the target itself.
  const tAnswerMs = nFuture > 0 ? predicted[predicted.length - 1] : Date.parse(c.target);
  const tAnswer = new Date(tAnswerMs).toISOString();
  const xAnswer = logTime(release, tAnswer);
  const dx = Math.max(0, xAnswer - xLast);
  const horizonAnswerDays = Math.max(0, (tAnswerMs - Date.parse(lastAt)) / 86_400_000);

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

  // Per-tier new-joiner propensity ratio: measured at the last observed refresh
  // when there is one, else the calibrated frozen-history prior.
  const fTier = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    if (fTierObserved && fTierObserved.has(tier)) {
      fTier.set(tier, fTierObserved.get(tier) as number);
    } else if (fTierObserved) {
      // Tier absent from the observed refresh: fall back to the global mean.
      const all = [...fTierObserved.values()];
      fTier.set(tier, all.reduce((a, b) => a + b, 0) / all.length);
    } else {
      fTier.set(tier, F_PRIOR_BASE * (TIER_F_MULT[tier] ?? 1));
    }
  }

  const spreadBase = SPREAD_BASE_A * Math.sqrt(horizonAnswerDays) + SPREAD_BASE_B;
  // Extra refresh variance in the trend branch: each future refresh's keep is
  // uncertain by `dispersion` (logit), compounding over the predicted refreshes.
  const refreshVar = Math.min(REFRESH_VAR_CAP, dispersion * dispersion * nFuture);

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
      // Frozen page: no trend is identified. Exactly one refresh inside the
      // horizon, of the published size (players tracked x REFRESH_N_RATIO).
      // Count space: the trophy's earners grow by p * (new joiners), with
      // p = earnProb(f_tier, lastRate); the 100% trophy stays 100%.
      if (f.lastRate >= 100) {
        mu = logit(99.99);
      } else {
        const nNew = nLast * REFRESH_N_RATIO;
        const dN = Math.max(0, nNew - nLast);
        const p = earnProb(fTier.get(t.tier) ?? F_PRIOR_BASE, f.lastRate);
        const kNew = (f.lastRate / 100) * nLast + dN * p;
        const rateNew = (kNew / nNew) * 100;
        mu = logit(rateNew);
      }
      sigma = SIGMA_FROZEN;
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
      // Evaluated at the last predicted refresh, not at the target: the page
      // does not move between refreshes, so the answering read shows the
      // state as of that refresh.
      mu = f.lastLogit + slope * dx;
      const move = mu - f.lastLogit;
      if (move > MAX_LOGIT_MOVE) mu = f.lastLogit + MAX_LOGIT_MOVE;
      if (move < -MAX_LOGIT_MOVE) mu = f.lastLogit - MAX_LOGIT_MOVE;
      const dilVar = Math.pow(SPREAD_DILUTION * logNGrowth * elasticity, 2);
      sigma = Math.max(
        MIN_SPREAD,
        Math.sqrt(
          Math.pow(spreadBase, 2) + Math.pow(se * dx, 2) + dilVar + countVar + refreshVar,
        ),
      );
    }
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
