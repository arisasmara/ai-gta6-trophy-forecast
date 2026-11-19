# ai-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arisasmara/ai-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arisasmara/ai-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch (26 November 2026), as five quantiles per trophy.

## How it forecasts

Earn rates are not persistent — they dilute. The tracked player base grows through launch week while the players who have earned each trophy grow slower, so a trophy's rate falls as new, less progressed players join. `main.ts` therefore fits, per trophy, the trend of its logit earn rate against log-time since release (recency-weighted least squares, shrunk toward the tier mean) and extrapolates that trend to the target. Where a trophy's history is frozen — the launch-night page before Exophase's first refresh — the trend is unidentified, so it falls back to a dilution prior: one refresh inside the horizon keeping about a fifth of the rate, scaled by an elasticity that is 0 for a 100% trophy and near 1 for a rare one. Spreads are honest: a base term growing with the horizon, the slope's standard error times the extrapolation distance, a dilution term from the forecast growth of players tracked, and binomial count noise from the number of earners behind the last rate. Node 22, standard library only, no network, fully deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=pw8p8er1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
