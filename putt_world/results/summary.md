# Results summary

Three model seeds per course and 20 episodes per seed. Seeds are the replication unit.

## Planner success (mean ± SD over seeds; shots when sunk in brackets)

| Planner | open | gate | slalom | pocket |
|---|---:|---:|---:|---:|
| random | 0.13 ± 0.10 (7.1) | 0.13 ± 0.15 (4.6) | 0.15 ± 0.05 (3.9) | 0.07 ± 0.03 (5.0) |
| aim | 1.00 ± 0.00 (1.4) | 0.47 ± 0.16 (1.2) | 0.27 ± 0.13 (1.1) | 0.10 ± 0.05 (1.2) |
| true physics, straight cost | 1.00 ± 0.00 (1.4) | 0.73 ± 0.10 (2.0) | 0.35 ± 0.13 (2.0) | 0.40 ± 0.26 (3.0) |
| true physics, geodesic cost | 1.00 ± 0.00 (1.4) | 1.00 ± 0.00 (2.3) | 1.00 ± 0.00 (2.7) | 1.00 ± 0.00 (2.5) |
| latent L2, H=1 | 0.93 ± 0.12 (1.9) | 1.00 ± 0.00 (2.2) | 0.97 ± 0.06 (3.4) | 0.48 ± 0.06 (3.2) |
| latent L2, H=2 | 0.47 ± 0.29 (5.4) | 0.42 ± 0.38 (5.2) | 0.55 ± 0.09 (5.9) | 0.45 ± 0.13 (4.3) |
| latent L2, H=3 | 0.20 ± 0.05 (7.0) | 0.17 ± 0.08 (4.5) | 0.38 ± 0.15 (7.7) | 0.32 ± 0.26 (6.2) |
| latent L2 anytime, H=3 | 0.95 ± 0.09 (2.2) | 1.00 ± 0.00 (3.2) | 0.98 ± 0.03 (4.4) | 0.28 ± 0.10 (4.5) |
| neighbour graph, H=1 | 0.95 ± 0.05 (1.6) | 1.00 ± 0.00 (2.2) | 1.00 ± 0.00 (3.1) | 0.48 ± 0.34 (4.1) |
| shot graph replay, H=1 | 0.62 ± 0.33 (1.4) | 0.87 ± 0.19 (2.3) | 0.98 ± 0.03 (3.7) | 0.57 ± 0.20 (3.2) |
| shot graph model, H=1 | 0.60 ± 0.35 (1.3) | 0.82 ± 0.20 (2.4) | 1.00 ± 0.00 (3.6) | 0.55 ± 0.23 (4.0) |
| shot graph + L2 finish (post hoc) | 0.92 ± 0.10 (2.1) | 1.00 ± 0.00 (2.1) | 1.00 ± 0.00 (3.4) | 0.40 ± 0.26 (2.4) |

## Latent geometry

| Course | Spearman vs path length | Spearman vs straight line | Near-hole Spearman | Wall gap | Probe RMSE |
|---|---:|---:|---:|---:|---:|
| open | 0.93 ± 0.03 | 0.93 ± 0.03 | 0.81 ± 0.08 | n/a | 0.01 ± 0.00 |
| gate | 0.79 ± 0.01 | 0.77 ± 0.02 | 0.70 ± 0.07 | 7.88 ± 0.99 | 0.02 ± 0.00 |
| slalom | 0.82 ± 0.04 | 0.77 ± 0.06 | 0.63 ± 0.08 | 8.33 ± 0.11 | 0.02 ± 0.00 |
| pocket | 0.79 ± 0.01 | 0.72 ± 0.04 | 0.41 ± 0.13 | 6.99 ± 0.51 | 0.03 ± 0.00 |

## Ablations

| Setting | Value | Course | Seeds | Wall gap | Spearman vs path length | Success | Probe RMSE |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 1 | gate | 3 | 7.88 ± 0.99 | 0.79 ± 0.01 | 1.00 ± 0.00 | 0.02 ± 0.00 |
| baseline | 1 | slalom | 3 | 8.33 ± 0.11 | 0.82 ± 0.04 | 0.98 ± 0.03 | 0.02 ± 0.00 |
| power_scale | 0.25 | gate | 3 | 1.93 ± 0.15 | 0.17 ± 0.02 | 0.17 ± 0.08 | 0.02 ± 0.00 |
| power_scale | 0.25 | slalom | 3 | 2.05 ± 0.10 | 0.22 ± 0.01 | 0.12 ± 0.08 | 0.02 ± 0.00 |
| power_scale | 0.5 | gate | 3 | 4.04 ± 0.31 | 0.51 ± 0.01 | 0.37 ± 0.13 | 0.01 ± 0.00 |
| power_scale | 0.5 | slalom | 3 | 5.05 ± 0.10 | 0.44 ± 0.03 | 0.35 ± 0.13 | 0.03 ± 0.00 |
| transitions | 5000 | gate | 2 | 6.72 ± 1.91 | 0.66 ± 0.02 | 0.93 ± 0.11 | 0.07 ± 0.01 |
| transitions | 20000 | gate | 2 | 8.05 ± 1.43 | 0.79 ± 0.00 | 1.00 ± 0.00 | 0.02 ± 0.01 |
| weight | 0.01 | gate | 2 | 12.85 ± 17.47 | 0.34 ± 0.80 | 0.50 ± 0.71 | 0.13 ± 0.13 |
| weight | 1 | gate | 2 | 2.57 ± 0.11 | 0.54 ± 0.03 | 0.95 ± 0.00 | 0.02 ± 0.00 |

## Wall controls

| Variant | Seeds | Wall gap |
|---|---:|---:|
| real wall | 3 | 7.88 ± 0.99 |
| dead wall | 3 | 3.82 ± 0.22 |
| ghost wall | 3 | 0.96 ± 0.01 |

## Dynamics and cost swapped (H=1)

| Dynamics, cost | open | gate | slalom | pocket |
|---|---:|---:|---:|---:|
| physics, latent L2 | 1.00 ± 0.00 | 0.95 ± 0.09 | 0.88 ± 0.13 | 1.00 ± 0.00 |
| physics, shot graph | 0.72 ± 0.26 | 0.93 ± 0.12 | 0.98 ± 0.03 | 0.92 ± 0.10 |
| predictor, latent L2 | 0.88 ± 0.16 | 1.00 ± 0.00 | 0.98 ± 0.03 | 0.50 ± 0.18 |
| predictor, path length | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.88 ± 0.20 | 0.53 ± 0.28 |

## Prediction error and rank

| Course | Error, wall shots | Error, other shots | Effective rank |
|---|---:|---:|---:|
| open | nan | 0.031 | 6.2 |
| gate | 0.052 | 0.044 | 6.8 |
| slalom | 0.059 | 0.054 | 7.3 |
| pocket | 0.074 | 0.041 | 7.1 |

## Share of random shots changed by a wall

| Course | Power | Share |
|---|---:|---:|
| gate | 1 | 0.228 |
| gate | 0.5 | 0.068 |
| gate | 0.25 | 0.017 |
| slalom | 1 | 0.415 |
| slalom | 0.5 | 0.134 |
| slalom | 0.25 | 0.035 |
