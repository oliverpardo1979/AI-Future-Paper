# Empirical motivation data

This folder contains the small, source-specific datasets used to build the two
empirical-motivation figures in the paper. They motivate the scale and direction
of the mechanisms in the model; they are not used to calibrate either CES
elasticity.

## Files

- `ai_production_2023_2025.csv`: U.S. AI compute spending, raw compute capacity,
  and quality-adjusted inference and training indices from Tables 1 and 2 of
  Korinek and McKelvey (2026). The quality-adjusted indices set 2023 equal to one
  and compound the reported annual percentage growth rates.
- `data_center_energy.csv`: global data-center electricity consumption for 2024
  and 2025 and the IEA central projection for 2030. The file explicitly marks the
  projected observation.
- `metr_time_horizon_1_1.csv`: the public METR Time Horizon 1.1 estimates
  downloaded on 12 August 2026. The horizon is measured in minutes of human-expert
  task-completion time. METR cautions that estimates above 16 hours are unreliable
  with the current task suite.
- `anthropic_directive_share.csv`: the share of sampled Claude.ai conversations
  classified as directive in three Economic Index vintages. Directive use is a
  subset of automation, not the full automation share.

## Primary sources

- https://www.bankofcanada.ca/2026/06/staff-working-paper-2026-20/
- https://www.iea.org/reports/energy-and-ai/executive-summary
- https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary
- https://metr.org/time-horizons/
- https://metr.org/assets/benchmark_results_1_1.yaml
- https://www.anthropic.com/research/anthropic-economic-index-september-2025-report
- https://www.anthropic.com/research/anthropic-economic-index-january-2026-report

