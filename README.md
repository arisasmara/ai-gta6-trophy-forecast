# ai-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arisasmara/ai-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arisasmara/ai-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch (26 November 2026), as five quantiles per trophy.

## How it forecasts

Earn rates are not persistent — they dilute, and they dilute in discrete steps. Exophase refreshes a game's counts on its own schedule: the page sits unchanged between refreshes and jumps at each one (638 players tracked, every rate frozen, then a refresh to 3,868 players with every rate but the 100% one lower). `main.ts` therefore works in count space: it detects refreshes from jumps in players tracked, measures each tier's keep factor and new-joiner earn propensity at the most recent observed refresh, predicts future refresh times from the observed cadence, and evaluates the forecast as of the last predicted refresh before the target — because the page does not move after it. Where a trophy's history is frozen — the launch-night page before Exophase's first refresh — no refresh has been observed, so it falls back to a dilution prior for exactly one refresh of the published size (3,868 / 638 players): each trophy's earners grow by p * (new joiners), with p = f * rate and f = 0.04 per tier (platinum 0.6x, gold 0.8x, silver 1.0x, bronze 1.2x), rising toward 1 as a rate approaches 100%. Where history varies, the per-trophy recency-weighted logit trend against log-time since release (shrunk toward the tier mean) is the base, evaluated at the last predicted refresh, with a spread that adds a term for refresh uncertainty. Spreads are honest: a base term growing with the horizon, the slope's standard error times the extrapolation distance, a dilution term from the forecast growth of players tracked, and binomial count noise from the number of earners behind the last rate. Node 22, standard library only, no network, fully deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=pw8p8er1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
