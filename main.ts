// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// A per-market quantile forecaster in logit space. Prediction-market prices
// are step-like: flat for minutes, then a tick, now and then a jump, and a
// market sitting at 0.01 can double while one at 0.5 cannot. So instead of
// the house baseline's symmetric random walk in price space, this entry
// models each market's logit(price) as a random walk with no drift:
//
//   q_tau(h) = sigmoid(logit(x) + z_tau * sigma * sqrt(h))
//
// where x is the price at the checkpoint, sigma is that market's own
// per-minute logit volatility, and z_tau is the standard normal quantile.
// The logit transform makes the intervals asymmetric near 0 and 1 (a
// 0.01 market's upside is roomier than its downside) and turns the
// occasional 20-point jump into an ordinary logit move.
//
// sigma is estimated jump-robustly from the last 720 minutes of one-minute
// logit changes: max(MAD, half the 3x-MAD-trimmed RMS), floored at 0.015
// and capped at 0.6 logit units per square-root minute. The floor matters:
// most minutes a market does not move at all, so the raw estimate would
// collapse to ~0 and the forecast would claim a certainty the market does
// not have. The outer quantiles (|z| > 1) are widened by 1.5x, because
// jumps put more mass in the tails than a Gaussian allows.
//
// No drift: on the practice cases, any shrunk recent-trend drift (either
// sign) and any pull toward the 24-hour mean made the loss worse, so the
// median stays at the checkpoint price.
//
// Calibrated on the six practice cases (scored against the next case's
// prices, which cover each 4-hour window): pinball loss 0.323 on p00-p04
// vs 0.393 for the house baseline, better on every case, with 90%
// coverage 0.80. The practice is another event's instance, so this is a
// sanity check, not a promise about the final.
//
// Deterministic: no clock, no randomness, no network, stdlib only.
// Reads every <id>.in in the input directory (the grader runs one case
// at a time) and writes <id>.out per case to the output directory.
// MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output
// outside the grader's container.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** How much wider the outer (5% / 95%) quantiles are than the Gaussian's. */
const TAIL_WIDEN = 1.5;
/** Minutes of recent history used to estimate a market's volatility. */
const SIGMA_WINDOW = 720;
/** Logit-volatility floor and cap, in logit units per square-root minute. */
const SIGMA_FLOOR = 0.015;
const SIGMA_CAP = 0.6;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0, 1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** log(p / (1 - p)), with p kept off the exact 0 / 1 edges. */
function logit(p: number): number {
  const c = Math.min(0.999999, Math.max(0.000001, p));
  return Math.log(c / (1 - c));
}

/** 1 / (1 + exp(-v)), safe for large |v|. */
function sigmoid(v: number): number {
  if (v > 30) return 1;
  if (v < -30) return 0;
  return 1 / (1 + Math.exp(-v));
}

/**
 * A market's per-minute logit volatility from its recent one-minute logit
 * changes: the median absolute change (robust to jumps and to flat
 * stretches) blended with half the trimmed RMS, so a market that really
 * does move gets a wider band than a flat one, but one jump does not
 * blow the estimate up. Floored and capped (see the header comment).
 */
function estimateSigma(prices: number[]): number {
  const start = Math.max(0, prices.length - SIGMA_WINDOW - 1);
  const window = prices.slice(start);
  if (window.length < 2) return SIGMA_FLOOR;
  const changes: number[] = [];
  for (let i = 1; i < window.length; i++) changes.push(logit(window[i]) - logit(window[i - 1]));
  const abs = changes.map((d) => Math.abs(d)).sort((a, b) => a - b);
  const mad = abs[Math.floor(abs.length / 2)] / 0.6745;
  const cap = Math.max(3 * mad, 1e-9);
  let sq = 0;
  for (const d of changes) {
    const t = Math.min(Math.abs(d), cap);
    sq += t * t;
  }
  const trimmedRms = Math.sqrt(sq / changes.length);
  const sigma = Math.max(mad, 0.5 * trimmedRms);
  return Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  // Drop non-finite prices (a NaN would poison the logit and the whole
  // case); carry the last good price forward, as the grid itself does.
  const prices: number[] = [];
  for (const p of m.prices) {
    if (Number.isFinite(p)) prices.push(p);
    else if (prices.length > 0) prices.push(prices[prices.length - 1] as number);
  }
  if (prices.length === 0) {
    // No history at all: a wide, honest prior rather than a crash, so the
    // case stays valid. (The kit's cases always carry prices.)
    return Array.from({ length: horizon }, () => [0, 0.25, 0.5, 0.75, 1]);
  }
  const x = prices[prices.length - 1] as number;
  const lx = logit(x);
  const sigma = estimateSigma(prices);
  const zz = Z.map((z) => (Math.abs(z) > 1 ? z * TAIL_WIDEN : z));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = sigma * Math.sqrt(h);
    rows.push(zz.map((z) => price6(sigmoid(lx + z * spread))));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
