# EGPU
This repository contains the newspaper dataset, keywords and related codes designed in the paper.

### Environmental Governance Policy Uncertainty Index: Overview

The Environmental Governance Policy Uncertainty (EGPU) Index quantifies uncertainty in environmental policy within the Chinese context, drawing methodological inspiration from "Measuring Economic Policy Uncertainty" by Baker, Bloom, and Davis (2016). This index captures the frequency and intensity of discussions surrounding environmental policy uncertainty in major Chinese newspapers, providing a time-series measure of policy-related uncertainty from 2000 to 2023.

**Methodology**: Following Baker et al. (2016), our approach employs a text-based analysis of articles from five prominent Chinese newspapers: *Guangming Daily*, *Economic Daily*, *Southern Weekend*, *People's Daily*, and *China Youth Daily*. We identify articles that simultaneously mention terms related to (1) uncertainty (e.g., "不确定", "难以预测"), (2) policy (e.g., "生态环境部", "法律"), and (3) environmental issues (e.g., "气候变化", "碳排放"). The raw counts are weighted by newspaper circulation to account for varying publication volumes, then aggregated at a daily frequency. To ensure robustness, we apply era-specific min-max normalization across three distinct periods (2000–2015, 2016–2020, 2021–2023), reflecting key policy milestones such as the Paris Agreement and China's carbon neutrality pledge. The resulting index is scaled to a 0–100 range and smoothed using a 45-day moving average to highlight trends.

**Applications**: The EGPU Index serves as a valuable tool for researchers, policymakers, and analysts studying the interplay between environmental policy uncertainty and economic, social, or environmental outcomes in China. By quantifying uncertainty dynamics, it facilitates investigations into how policy ambiguity influences investment decisions, market behavior, and sustainability initiatives.
