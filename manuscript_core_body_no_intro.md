# Core Manuscript Text Excluding Introduction

## Methods

### Overview of the frontier-based resilience sizing framework

We developed a frontier-based PV-BESS assessment framework to quantify how installed storage capacity affects outage resilience for a building community. The framework uses hourly building load, PV generation, tariff, outage, and techno-economic inputs to evaluate candidate PV-BESS designs under stochastic grid interruptions. PV and battery capacities are swept to construct a resilience cost-reliability frontier.

The framework is designed to make the resilience trade-off explicit rather than to report a single optimum. For each capacity pair, the model estimates unserved energy (UE), unserved-energy cost, outage load served ratio, mean scenario loss-of-load probability (LOLP), chance-constraint probability, and annual system total cost. These outputs allow the minimum-total-cost design to be compared with a reliability-constrained design that satisfies a target mean scenario LOLP.

### Electric-equivalent building load and PV generation

The building community load is represented as a full electric-equivalent demand. Lighting and equipment are treated as direct electrical loads, while cooling and heating thermal loads are converted to electrical-equivalent loads using the corresponding coefficients of performance. For each hour \(t\), the electric-equivalent load is

\[
L_t^{eq} = L_t^{electric} + \frac{L_t^{cooling}}{COP_c} + \frac{L_t^{heating}}{COP_h}.
\]

The outage service target in this study is the full electric-equivalent building load. We therefore do not introduce separate load-priority classes or a critical-load fraction in the present implementation. This choice makes the reported reliability metrics conservative for applications in which only a subset of building demand would need to be served during an outage.

PV generation is derived from the hourly generation profile of one square metre of module area and converted to generation per installed kW using the PV module power density. For a candidate installed PV capacity, the model scales the hourly PV profile to obtain total PV generation and checks the corresponding rooftop area requirement. This input-driven formulation allows the same workflow to be applied to other city-building cases when 8760 h load, PV, and tariff inputs are available.

### PV-BESS dispatch under grid-connected and outage conditions

For each PV-BESS candidate, hourly dispatch is simulated with PV generation, building demand, grid availability, tariff, and battery constraints. During normal grid-connected operation, PV generation can serve load, charge the battery, or reduce grid purchases according to the dispatch logic and time-of-use tariff. The battery is constrained by energy capacity, power capacity, charging and discharging limits, round-trip efficiency, minimum and maximum state of charge, and initial state of charge. During outage hours, the system uses available PV and battery discharge to serve the electric-equivalent building load; any remaining unmet demand is recorded as unserved energy.

The dispatch model therefore links two operational regimes that are often treated separately: normal tariff-driven operation and outage backup operation. This is important because the same battery capacity cannot be used independently for bill reduction and emergency backup service.

### Stochastic outage generation

Grid interruption scenarios are generated from annual interruption frequency and duration assumptions. For each scenario, outage events are sampled and converted into an hourly grid-availability profile, including outage fractions for hours that are only partially interrupted. The frontier uses 100 stochastic outage scenarios for the SG_UTown case. Each candidate PV-BESS design is evaluated across the same scenario set so that differences in resilience performance are driven by design choice rather than by different outage realizations.

The stochastic treatment is used because a single outage event cannot represent the dependence of PV-BESS performance on event timing. Outages occurring at night, during high-load periods, or after low-solar hours can lead to different resilience outcomes from outages occurring during periods of high PV availability. The scenario set therefore supports reliability metrics that summarize both average outage loss and scenario-to-scenario variability.

### Resilience and cost metrics

For each outage scenario, UE is calculated as the sum of unmet electric-equivalent load during outage periods. The monetary value of this outage loss is calculated by multiplying UE by the value of lost load (VoLL). The outage load served ratio is calculated as one minus the ratio of UE to the total outage-weighted electric-equivalent load.

Reliability is represented by two complementary metrics. The scenario LOLP is defined on an energy basis as UE divided by the total outage-weighted electric-equivalent building load. The mean scenario LOLP is the mean scenario LOLP across the 100 stochastic outage scenarios. The chance-constraint probability is the proportion of scenarios whose scenario LOLP is less than or equal to the selected LOLP threshold. In the SG_UTown case, the threshold is 0.10.

Annual system total cost combines annualized PV cost, annualized battery cost, O&M cost, grid energy cost, and expected unserved-energy cost. This cost definition allows additional battery capacity to have two opposing effects: it increases annualized storage cost, but it can reduce grid purchases and outage losses. The frontier analysis identifies both the minimum-total-cost design and the least-cost design satisfying the mean scenario LOLP threshold.

## Case Study

The framework was applied to the SG_UTown building-community case using input files for annual load, PV generation, time-of-use tariff, and techno-economic parameters. The load input contains 8760 hourly records for cooling, heating, lighting, and equipment demand. In this Singapore case, heating demand is zero, and cooling is converted to electric-equivalent demand using a cooling COP of 5. The PV input gives hourly generation per square metre of PV module area, and installed PV capacity is converted to module area using a module power density of 0.135 kW m-2. The available rooftop area is 300,000 m2.

The tariff input represents a 24 h time-of-use schedule repeated across the year, with a lower price of 0.239 SGD kWh-1 during 00:00-08:00 and 21:00-23:00 and a higher price of 0.329 SGD kWh-1 during 09:00-20:00. The outage model uses a baseline SAIFI of 3 events per year and SAIDI of 150 min per year. The value of lost load is 50 SGD kWh-1. PV CAPEX is set to 1200 SGD kW-1, battery energy CAPEX to 400 SGD kWh-1, and battery power CAPEX to 180 SGD kW-1. Battery operation is constrained by a maximum C-rate of 0.5 h-1, round-trip efficiency of 0.92, minimum SOC of 0.10, maximum SOC of 0.95, and initial SOC of 0.50.

The frontier evaluates PV and BESS candidates across 100 stochastic outage scenarios. Results are reported for full-building-load resilience, not for a reduced critical-load fraction.

## Results

### Building, PV and outage inputs define a time-dependent resilience problem

The SG_UTown input profiles show that outage resilience cannot be inferred from annual energy balance alone. The representative-week load, scaled PV generation, and tariff schedule in Figure 2 show that building demand, PV availability, and electricity price vary on different time scales. The outage scenarios in Figure 3 further show that interruption timing and duration vary across the stochastic samples. Because outage performance depends on the coincidence of load, PV generation, battery state of charge, and grid availability, the following analysis evaluates hourly dispatch rather than using an aggregate annual sizing rule.

The PV-building load mismatch provides the immediate reason for including storage in the resilience assessment. Figure 4 compares the distribution of full-building electric-equivalent load with PV generation under selected PV capacities. The mismatch indicates that PV capacity alone does not guarantee outage service, particularly when outages occur outside periods of high solar output. Battery capacity is therefore treated as the main controllable resource for shifting PV energy and supporting load during interruptions.

### Increasing battery capacity improves outage service, but with diminishing returns

Battery capacity increases the share of outage load served and reduces unserved energy, but the improvement is nonlinear. Figure 5 shows that outage load served ratio rises as battery capacity increases for the selected PV capacities. The response is strongest over the capacity range where additional storage can directly cover previously unmet outage demand. At higher storage capacities, the served-ratio curves flatten, indicating that additional capacity provides smaller incremental resilience gains because remaining unmet load is increasingly tied to event timing, PV availability, and state-of-charge conditions.

The same pattern appears when resilience is measured as physical and monetary outage loss. Figure 6 shows that mean UE and mean unserved-energy cost decrease as battery capacity increases. This result connects the technical reliability metric to the economic valuation of resilience: reducing unserved energy reduces the expected outage-loss term in the system cost. The cost reduction, however, must be compared with the additional annualized cost of installing and maintaining battery capacity.

### The minimum-cost design differs from the reliability-constrained design

The cost-reliability frontier shows that the least-cost capacity choice is not necessarily the capacity required to satisfy a reliability target. Figure 7 identifies a minimum-total-cost design of 20 MW PV and 10 MWh BESS, with an annual system total cost of 8.69 million SGD yr-1. This design has mean UE of 2101 kWh yr-1, mean outage load served ratio of 0.818, mean scenario LOLP of 0.193, and chance-constraint probability of 0.58. It is therefore the lowest-cost design among the evaluated candidates, but it does not satisfy the mean scenario LOLP threshold of 0.10.

Applying the reliability constraint shifts the selected design to a larger PV-BESS system. The least-cost candidate satisfying mean scenario LOLP less than or equal to 0.10 uses 28 MW PV and 20 MWh BESS. This reliability-constrained design reduces mean UE to 971 kWh yr-1, increases mean outage load served ratio to 0.910, reduces mean scenario LOLP to 0.097, and raises the chance-constraint probability to 0.75. Its annual system total cost is 9.01 million SGD yr-1. Thus, meeting the reliability target requires 8 MW more PV and 10 MWh more battery capacity than the minimum-total-cost design, increasing annual system cost by about 0.32 million SGD yr-1.

### LOLP and chance-constraint metrics reveal different aspects of reliability

Mean scenario LOLP and chance-constraint probability provide complementary views of outage performance. Figure 8 shows how mean scenario LOLP decreases as battery capacity increases, allowing candidate designs to be compared against the 0.10 reliability threshold. This metric summarizes the average unserved-energy share of outage-weighted building load across the stochastic scenarios. It is useful for identifying whether the mean reliability target is met, but it does not show how consistently that target is achieved across scenarios.

Figure 9 addresses this scenario-level consistency by reporting the chance-constraint probability, defined as the share of outage scenarios whose scenario LOLP is less than or equal to 0.10. The minimum-total-cost design satisfies this condition in 58% of scenarios, while the reliability-constrained design satisfies it in 75% of scenarios. This difference shows that reducing mean scenario LOLP improves reliability on average, but stochastic outage timing still leaves a fraction of scenarios in which the LOLP threshold is not met.

### Marginal reliability improvement has an economic structure

The marginal-cost analysis shows that reliability improvement is not uniformly expensive across the design space. Figure 10 relates changes in system cost to changes in mean scenario LOLP and chance-constraint probability. In capacity regions where additional storage avoids substantial outage loss or grid purchases, reliability improvement can be economically attractive. Beyond the effective improvement region, additional capacity produces smaller reductions in UE or LOLP while still adding annualized battery cost. The frontier therefore supports a decision interpretation: the relevant question is not whether larger batteries are more reliable, but where the marginal cost of additional reliability becomes difficult to justify for the selected reliability target.

## Discussion

This study shows that PV-BESS resilience planning for a building community is best interpreted as a cost-reliability frontier rather than as a single cost-minimization problem. In the SG_UTown case, the minimum-total-cost design and the reliability-constrained design are different points on the frontier. The minimum-total-cost design is economically attractive under the evaluated assumptions, but its mean scenario LOLP of 0.193 exceeds the 0.10 threshold. The reliability-constrained design meets the threshold with larger PV and battery capacities, reducing UE and improving scenario-level reliability at a higher annual system cost.

The results also show why UE, mean scenario LOLP, and chance-constraint probability should be reported together. UE captures the physical quantity of unmet load and directly determines outage-loss cost through VoLL. Mean scenario LOLP summarizes average reliability across stochastic outage scenarios. Chance-constraint probability indicates how often the reliability target is satisfied across individual scenarios. A design can improve one of these metrics without fully resolving the others, especially when outage timing interacts with PV availability and battery state of charge.

Several boundaries should be considered when interpreting these results. First, the present study uses the full electric-equivalent building load as the outage service target. This is conservative relative to cases in which only critical loads must be maintained, but it does not distinguish between load priorities. Second, the reported numerical results are based on the SG_UTown input data, tariff, outage assumptions, and cost parameters. Other building communities with different load shapes, PV resources, tariff structures, outage risks, or VoLL assumptions may occupy different regions of the frontier. Third, battery-cost and SAIDI-VoLL two-dimensional sensitivity figures are not included in the current reported output, so conclusions about those broader parameter spaces are left for future analysis.

The framework is nevertheless transferable because its inputs are modular: hourly load, PV generation, tariff, outage parameters, cost assumptions, and battery constraints can be replaced for other cases. The main practical implication is that resilience investment should be evaluated using both economic and reliability criteria. A planner interested only in minimum expected cost may select a smaller system than a planner required to satisfy a reliability threshold.

## Conclusions

This work establishes a frontier-based PV-BESS assessment framework for building-community outage resilience. The framework constructs a resilience cost-reliability frontier by evaluating PV and battery capacity candidates across stochastic outage scenarios.

Applied to the SG_UTown case, the framework identifies a minimum-total-cost design of 20 MW PV and 10 MWh BESS with an annual system total cost of 8.69 million SGD yr-1, mean scenario LOLP of 0.193, and chance-constraint probability of 0.58. A larger reliability-constrained design of 28 MW PV and 20 MWh BESS satisfies the 0.10 mean scenario LOLP threshold, reducing mean scenario LOLP to 0.097 and increasing chance-constraint probability to 0.75 at an annual system total cost of 9.01 million SGD yr-1.

Overall, the results demonstrate that PV-BESS resilience planning requires joint reporting of cost, unserved energy, average reliability, and scenario-level reliability. The current case study is bounded to the SG_UTown data and full-building-load service target. Extending the framework to critical-load segmentation, additional cities, and broader outage-value sensitivity would further clarify how local risk and operational priorities shape cost-effective resilience investment.

## Claim-evidence map

| Claim | Evidence used in draft | Status |
|---|---|---|
| Battery capacity improves outage service with diminishing returns. | Figures 5 and 6; frontier candidate metrics. | Supported by existing outputs. |
| Minimum-cost and reliability-constrained designs differ. | `SG_UTown_Frontier_Optimal_Designs.csv`: 20 MW + 10 MWh versus 28 MW + 20 MWh. | Supported by existing outputs. |
| Chance-constraint probability adds scenario-level reliability information beyond mean scenario LOLP. | Figure 9 and optimal-design values of 0.58 versus 0.75. | Supported by existing outputs. |
| Battery-cost and SAIDI-VoLL sensitivity should not be claimed as current results. | Figure 11/12 outputs are absent from current output directory. | Marked as future extension. |
| Results generalize quantitatively beyond SG_UTown. | No multi-city or broader sensitivity output provided. | Not claimed. |
