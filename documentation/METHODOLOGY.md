# fleet2 — Model Methodology

## Contents

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [Vehicle Model](#3-vehicle-model)
   - 3.1 [Mass](#31-mass)
   - 3.2 [Fuel Consumption](#32-fuel-consumption)
   - 3.3 [Range](#33-range)
   - 3.4 [Annual Distance](#34-annual-distance)
   - 3.5 [Fuel Cell Stack Replacements](#35-fuel-cell-stack-replacements)
   - 3.6 [Emissions](#36-emissions)
   - 3.7 [Capital Cost](#37-capital-cost)
   - 3.8 [Annual Cost and Revenue](#38-annual-cost-and-revenue)
   - 3.9 [NPV and TCO](#39-npv-and-tco)
4. [Fleet Dynamics](#4-fleet-dynamics)
   - 4.1 [Activity Requirement](#41-activity-requirement)
   - 4.2 [Initial Stock](#42-initial-stock)
   - 4.3 [Year-by-Year Simulation](#43-year-by-year-simulation)
5. [Market Share](#5-market-share)
   - 5.1 [Nested Logit](#51-nested-logit)
   - 5.2 [Production Cap](#52-production-cap)
   - 5.3 [Shadow Pricing](#53-shadow-pricing)
6. [Policy Instruments](#6-policy-instruments)
   - 6.1 [Carbon Tax](#61-carbon-tax)
   - 6.2 [Low Carbon Fuel Standard (LCFS)](#62-low-carbon-fuel-standard-lcfs)
   - 6.3 [ZEV Mandate](#63-zev-mandate)
   - 6.4 [GVWL Exemption](#64-gvwl-exemption)
7. [Monte Carlo Framework](#7-monte-carlo-framework)
   - 7.1 [Parameter Distributions](#71-parameter-distributions)
   - 7.2 [Grouped Sampling](#72-grouped-sampling)
   - 7.3 [Convergence](#73-convergence)
   - 7.4 [Scenarios](#74-scenarios)
8. [Outputs](#8-outputs)

---

<details>
<summary><strong>1. Overview</strong></summary>

A year-by-year fleet adoption simulation for Class 8 heavy-duty trucks (HDTs) in British Columbia, Canada, covering 2025--2050. The simulation tracks three vehicle types (sleeper, day cab, straight truck) across seven powertrains (diesel ICE, mild hybrid, plug-in hybrid, battery electric, fuel cell, hydrogen ICE hybrid, dual-fuel H2/diesel) and seven fuel types.

The model operates in three layers: (1) a **vehicle physics layer** that computes mass, fuel consumption, range, annual distance, emissions, and costs for each cohort (vehicle type, powertrain, model year); (2) a **fleet dynamics layer** that rolls surviving cohorts forward year by year and sizes new sales to meet an exogenous activity target; and (3) a **market share layer** that allocates new sales across powertrains via a nested logit with production constraints enforced through shadow pricing. Four policy instruments can be applied individually or in combination. A Monte Carlo wrapper propagates parametric uncertainty through all layers simultaneously.

The simulation time step is one year. Each vehicle cohort is indexed by vehicle type $k$, powertrain $p$, and model year $y$; age $a = t - y$ tracks how old a cohort is in calendar year $t$.

</details>

---

<details>
<summary><strong>2. Notation</strong></summary>

All symbols are defined here and used consistently throughout. Symbols are subscripted to denote the sets they belong to and superscripted to provide context.

**Cost convention.** Uppercase $C$ denotes a dollar amount (\$). Lowercase $c$ denotes a per-unit rate (\$/km, \$/kWh, \$/tCO2e, etc.) that must be multiplied by a quantity to obtain dollars.

### Sets and Indices

| Symbol | Description |
|--------|-------------|
| $t \in T$ | Calendar year, $T = \{2025, \ldots, 2050\}$ |
| $y \in Y$ | Vehicle registration (model) year, $Y = \{2000, \ldots, 2050\}$ |
| $a \in A$ | Vehicle age in years, $A = \{0, \ldots, 24\}$; $a = t - y$ |
| $k \in K$ | Vehicle type: sleeper, day\_cab, straight |
| $p \in P$ | Powertrain: dice, he, phe, be, fc, hice, dhice |
| $f \in F$ | Fuel: diesel, H2 (electrolysis / pyrolysis / electrified pyrolysis), electricity (depot slow charge / fast charge / public non-depot slow charge, e.g. sleeper PHE charging during a rest break) |
| $d$ | Drive cycle label assigned to age $a$: long\_haul, regional\_haul, or short\_haul |

### Fleet Counts

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $N_{k,p,y,t}$ | vehicles | Stock of type $k$, powertrain $p$, model year $y$, surviving in year $t$ | 4.2 |
| $\chi_{k,a}$ | -- | Survival rate at age $a$ for type $k$ (fraction of cohort still on road) | 4.2 |

### Vehicle Mass

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $M^{\text{frame}}_k$ | kg | Frame mass of vehicle type $k$ | 3.1 |
| $M^{\text{trailer}}_k$ | kg | Trailer mass | 3.1 |
| $M^{\text{conv}}_c$ | kg | Mass of converter or transmission component $c$ | 3.1 |
| $M^{\text{spec}}_c$ | kg/unit | Specific mass of energy storage component $c$ per unit of capacity | 3.1 |
| $X_c$ | unit | Capacity of energy storage component $c$ (kWh, kg H2, or L diesel) | 3.1 |
| $M^{\text{unladen}}_{k,p}$ | kg | Unloaded vehicle mass (sum of all component masses) | 3.1 |
| $\bar{M}^{\text{unladen}}_k$ | kg | Reference unloaded mass for the diesel variant of type $k$ | 3.1 |
| $M^{\text{GVWL}}_k$ | kg | Gross vehicle weight limit for type $k$ | 3.1 |
| $\Delta M^{\text{GVWL}}_k$ | kg | Additional GVWL headroom granted to ZEVs under the GVWL exemption | 3.1, 6.4 |
| $\omega_k$ | -- | Fraction of loads that are weight-limited | 3.1 |
| $\nu_{k,p}$ | -- | Payload fraction: 1 = no penalty; $<1$ = heavier drivetrain displaces payload | 3.1 |
| $\bar{M}^{\text{payload}}_{k,d}$ | kg | Reference payload for drive cycle $d$ and vehicle type $k$ | 3.1 |
| $M^{\text{payload}}_{k,p,a}$ | kg | Payload mass at age $a$ | 3.1 |
| $M^{\text{total}}_{k,p,a}$ | kg | Total laden vehicle mass at age $a$ | 3.1 |

### Fuel Consumption

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $v_{k,a}$ | km/h | Average driving speed at age $a$ for type $k$ | 3.4 |
| $\mu$ | -- | Aerodynamic drag coefficient | 3.2 |
| $P^{\text{acc}}$ | W | Accessory electrical load | 3.2 |
| $\eta_p$ | -- | Peak efficiency of the primary energy converter for powertrain $p$ | 3.2 |
| $\eta^{\text{fill}}_f$ | -- | Refuelling or charging efficiency for fuel $f$ (tank/battery side to source side) | 3.2 |
| $H_f$ | J/unit | Lower heating value (LHV) of fuel $f$ | 3.2 |
| $\phi_f$ | -- | Declared proportion of total energy supplied by fuel $f$ (multi-fuel powertrains) | 3.2 |
| $\beta_0,\,\beta_i,\,\beta_{ij}$ | varies | Surrogate polynomial coefficients: intercept, linear, and pairwise-interaction terms | 3.2 |
| $\hat{x}_{p,d}$ | unit/km | Surrogate output: fuel consumption for powertrain $p$ on drive cycle $d$ (tank-side units) | 3.2 |
| $x_{f,a}$ | unit/km | Fuel consumption of fuel $f$ at age $a$ (tank-side units per km) | 3.2 |
| $\tilde{x}_{f,a}$ | unit/km | Fuel consumption in source units per km, after refuelling efficiency adjustment | 3.2 |

### Range and Annual Distance

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $u_c$ | -- | Usable capacity fraction of energy storage component $c$ | 3.3 |
| $r$ | unit/h | Refuelling or recharging rate of the binding energy storage component | 3.3 |
| $d^{\text{max}}_{c,a}$ | km | Range provided by energy storage component $c$ at age $a$ | 3.3 |
| $d^{\text{max}}_a$ | km | Effective vehicle range at age $a$ (minimum over all energy storage components) | 3.3 |
| $d^{\text{tgt}}_{k,a}$ | km/yr | Target annual distance at age $a$ for type $k$ | 3.4 |
| $d^{\text{daily}}_{k,a}$ | km/day | Daily working-day distance target | 3.4 |
| $d_{k,p,y,a}$ | km/yr | Actual annual distance achieved at age $a$ | 3.4 |
| $d^{\text{en}}_a$ | km/yr | Annual distance driven via en-route fast charging | 3.4 |
| $\xi$ | yr$^{-1}$ | Battery capacity fade per year of age | 3.4 |
| $\theta$ | cycle$^{-1}$ | Battery capacity fade per charge cycle | 3.4 |
| $N^{\text{cyc}}_a$ | cycles | Cumulative charge cycles accumulated to age $a$ | 3.4 |
| $Q_{f,a}$ | unit/yr | Annual consumption of fuel $f$ at age $a$ (source units) | 3.4 |

### Fuel Cell Replacements

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $L^{\text{fc}}$ | h | Fuel cell stack lifetime in operating hours | 3.5 |
| $r^{\text{fc}}_a$ | -- | Replacement indicator: 1 if stack is replaced at age $a$, else 0 | 3.5 |

### Emissions

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $i^{\text{supply}}_f$ | kgCO2e/unit | Supply-chain (well-to-tank) emissions intensity of fuel $f$ | 3.6 |
| $i^{\text{use}}_f$ | kgCO2e/unit | Tailpipe (tank-to-wheel) emissions intensity of fuel $f$ | 3.6 |
| $i^{\text{emb}}_c$ | kgCO2e/kg or kgCO2e/unit | Embodied emission factor for component $c$ (per kg of mass, or per unit of capacity for ESS) | 3.6 |
| $\zeta_k$ | -- | Number of trailers per truck for type $k$ | 3.6 |
| $\Gamma^{\text{emb}}_{k,p,y,a}$ | kgCO2e | Embodied emissions attributed to the vehicle at age $a$ | 3.6 |
| $\Gamma^{\text{supply}}_{k,p,y,a}$ | kgCO2e/yr | Annual supply-chain emissions at age $a$ | 3.6 |
| $\Gamma^{\text{use}}_{k,p,y,a}$ | kgCO2e/yr | Annual tailpipe emissions at age $a$ | 3.6 |

### Costs

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $\delta$ | -- | Annual discount rate | 3.9 |
| $\Delta_a$ | -- | Survival-weighted discount factor: $\chi_{k,a}\,/\,(1+\delta)^a$ | 3.9 |
| $C^{\text{capital}}_{k,p,y}$ | \$ | Total capital cost at purchase (model year $y$) | 3.7 |
| $C^{\text{frame}}_k$ | \$ | Base chassis cost for vehicle type $k$ | 3.7 |
| $c^{O\&M}_p$ | \$/km | Per-km O\&M cost rate for powertrain $p$ | 3.8 |
| $c_f(t)$ | \$/unit | Fuel price of fuel $f$ in calendar year $t$ | 3.8 |
| $c^{\text{driver}}_k$ | \$/km | Per-km driver wage for vehicle type $k$ | 3.8 |
| $c^{\text{fc,rep}}_p(t)$ | \$/kW | Fuel cell stack replacement cost rate in year $t$ | 3.8 |
| $\varsigma_k$ | \$/t-km | Revenue per tonne-km for type $k$ | 3.8 |
| $\text{NPV}_{k,p,y}$ | \$ | Net present value of the vehicle cohort | 3.9 |
| $\text{TCO}_{k,p,y}$ | \$ | Total (discounted) cost of ownership | 3.9 |

### Fleet Dynamics

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $t_0,\,t_{\text{end}}$ | yr | Simulation start (2025) and end (2050) years | 4 |
| $\mathcal{A}_k(t)$ | t-km/yr | Fleet activity requirement for type $k$ in year $t$ | 4.1 |
| $\mathcal{A}^{\text{tot}}_0$ | t-km/yr | Total fleet activity across all types in $t_0$ | 4.1 |
| $\alpha_k$ | -- | Activity share for vehicle type $k$ (fraction of total fleet activity) | 4.1 |
| $\alpha_{k,p,y,a}$ | t-km/yr | Activity per vehicle of cohort $(k,p,y)$ at age $a$ | 4.2 |
| $\kappa$ | -- | Annual fleet activity growth rate | 4.1 |
| $\Omega_k$ | t-km/vehicle | Survival- and growth-weighted lifetime activity per vehicle (initial stock denominator) | 4.2 |

### Market Share

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $S_{k,p,t}$ | -- | Market share of powertrain $p$ for type $k$ in year $t$ | 5.1 |
| $\lambda$ | \$$^{-1}$ | Logit price-sensitivity parameter (`price_lambda`) | 5.1 |
| $n \in \mathcal{T}$ | -- | Nest in the powertrain nesting tree: Liquid, Conventional, Hydrogen, Electric, or the root | 5.1 |
| $\ell_n$ | -- | Nest scale parameter (`nest_lambdas`); $\ell_n = 1$ collapses nest $n$ to a flat logit locally; root's scale is fixed at 1 | 5.1 |
| $V_p$ | -- | Leaf utility of powertrain $p$: $\lambda\,(\text{NPV}_{k,p,t} - \mu_p)$ | 5.1, 5.3 |
| $U_n$ | -- | Inclusive utility of nest $n$ (lambda-scaled log-sum-exp of its children) | 5.1 |
| $\bar{S}_{k,p,t}$ | -- | Production cap: maximum achievable market share in year $t$ | 5.2 |
| $S^{\text{init}}_{k,p}$ | -- | Initial market share floor for powertrain $p$ | 5.2 |
| $S^*$ | -- | Threshold share separating the nascent from the mature supply growth regime | 5.2 |
| $z^{\text{nac}}_{k,p}$ | -- | Nascent-phase annual supply growth rate (CAGR) for powertrain $p$ | 5.2 |
| $z^{\text{mat}}_{k,p}$ | -- | Mature-phase annual supply growth rate (CAGR) for powertrain $p$ | 5.2 |
| $\mu_p \geq 0$ | -- | Shadow cost applied to powertrain $p$'s NPV so its share respects $\bar{S}_{k,p,t}$ | 5.3 |

### Policy

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $\gamma$ | -- | Policy on/off indicator (1 = active, 0 = inactive) | 6 |
| $c^{\text{ct}}_t$ | \$/tCO2e | Carbon tax price in year $t$ | 6.1 |
| $c^{\text{lcfs}}$ | \$/tCO2e | LCFS credit price | 6.2 |
| $T^{\text{lcfs}}_t$ | -- | LCFS carbon intensity reduction target fraction in year $t$ | 6.2 |
| $\text{CI}^{\text{diesel}}$ | kgCO2e/L | Fixed reference diesel emissions intensity (supply + use) used only to anchor the LCFS standard -- distinct from, and not synced to, the model's own (blended) $i^{\text{supply}}_{\text{diesel}}$/$i^{\text{use}}_{\text{diesel}}$ used elsewhere | 6.2 |
| $\text{TCI}(t)$ | kgCO2e/J | Target carbon intensity per joule of fuel energy in year $t$ | 6.2 |
| $\text{EER}_p$ | -- | Energy Effectiveness Ratio for powertrain $p$'s primary non-diesel fuel; diesel fuel always uses EER $=1$ regardless of $p$ | 6.2 |
| $Z^{\text{tgt}}_t$ | -- | ZEV mandate target: required ZEV share of new sales in year $t$ | 6.3 |
| $p^{\text{zev}}$ | -- | Trial (bisection probe) ZEV share of new sales | 6.3 |
| $S^{\text{zev}}_t$ | -- | Realised ZEV share of new sales in year $t$ once the bisection converges | 6.3 |
| $c^{\text{credit}}(Z,p)$ | \$/credit | ZEV credit market price: a logistic function of the compliance gap $p^{\text{zev}} - Z$ | 6.3 |
| $c^{\text{penalty,max}}$ | \$/credit | Ceiling credit price (asymptote as the compliance gap $\to -\infty$) | 6.3 |
| $w$ | -- | Transition width: share-gap span over which $c^{\text{credit}}$ moves from $\sim$95\% to $\sim$5\% of $c^{\text{penalty,max}}$ | 6.3 |
| $\rho_k$ | credits/vehicle | Credits earned or owed per vehicle of type $k$ (`credits_per_vehicle`) | 6.3 |
| $\pi_t$ | -- | Payout rationing factor: fraction of full credit value actually paid to ZEVs in year $t$ | 6.3 |

### Monte Carlo

| Symbol | Units | Description | Sec. |
|--------|-------|-------------|------|
| $u$ | -- | Uniform cumulative probability draw, $u \in [0,1]$ | 7.1 |
| $\mathcal{G}$ | -- | Parameter group: set of parameters sharing a single draw $u$ | 7.2 |
| $N^{\text{runs}}$ | -- | Number of Monte Carlo runs | 7.2 |
| $N^{\text{cols}}$ | -- | Number of independent random draws per run | 7.2 |
| $D_{\text{KS}}$ | -- | Two-sample Kolmogorov--Smirnov statistic | 7.3 |
| $\tau_{\text{KS}}$ | -- | KS convergence tolerance | 7.3 |

</details>

---

<details>
<summary><strong>3. Vehicle Model</strong></summary>

One vehicle cohort is instantiated per $(k, p, y)$ combination. All quantities are computed once at instantiation and stored as age-arrays over $a \in A$.

### 3.1 Mass

**Component masses.** Converter and transmission components contribute a fixed mass $M^{\text{conv}}_c$. Energy storage components contribute specific mass times capacity:

$$M^{\text{unladen}}_{k,p} = M^{\text{frame}}_k + M^{\text{trailer}}_k + \sum_c M_c$$

where $M_c = M^{\text{conv}}_c$ for converters and transmissions, and $M_c = M^{\text{spec}}_c \cdot X_c$ for energy storage components.

**Payload penalty.** A heavier drivetrain displaces payload on weight-limited loads. The payload fraction is:

$$\nu_{k,p} = \max\!\left(0,\ 1 - \omega_k \left(1 - \frac{M^{\text{GVWL}}_k + \Delta M^{\text{GVWL}}_k - M^{\text{unladen}}_{k,p}}{M^{\text{GVWL}}_k - \bar{M}^{\text{unladen}}_k}\right)\right)$$

$\Delta M^{\text{GVWL}}_k = 0$ for non-ZEV powertrains; set by the GVWL exemption policy for ZEVs (Section 6.4). The ratio $(M^{\text{GVWL}}_k - M^{\text{unladen}}_{k,p})\,/\,(M^{\text{GVWL}}_k - \bar{M}^{\text{unladen}}_k)$ gives the remaining payload headroom relative to the diesel reference; $\omega_k$ scales the penalty by the fraction of loads that are weight-limited.

**Age-varying payload and total mass.** Payload depends on the drive cycle, which changes with vehicle age:

$$M^{\text{payload}}_{k,p,a} = \nu_{k,p} \cdot \bar{M}^{\text{payload}}_{k,d_a}$$

$$M^{\text{total}}_{k,p,a} = M^{\text{unladen}}_{k,p} + M^{\text{payload}}_{k,p,a}$$

---

### 3.2 Fuel Consumption

#### 3.2.1 Surrogate Training

FASTSim vehicle simulations are run over a design grid of vehicle parameter combinations $(M^{\text{total}},\,\mu,\,P^{\text{acc}},\,\eta_p)$ for each powertrain $p$ and drive cycle $d$. Each simulation produces a fuel consumption value in primary energy carrier units per km (L/km for diesel powertrains, kWh/km for battery-electric, kg/km for hydrogen). A degree-2 interaction polynomial is fitted to these simulation outputs via ordinary least squares:

$$\hat{x}_{p,d} = \beta_0 + \sum_i \beta_i \, z_i + \sum_{i \leq j} \beta_{ij}\, z_i z_j$$

where the feature vector is $\mathbf{z} = (M^{\text{total}},\ \mu,\ P^{\text{acc}},\ \eta_p^{-1})$. The inverse efficiency $\eta_p^{-1}$ is used because fuel consumption scales inversely with efficiency. Fit quality: $R^2 > 0.9999$ and MAPE $< 0.25\%$ across all surrogate models.

Five drive cycles are covered: long haul, regional haul, short haul, UDDS-HDT, and cruise-HDT. Surrogate mappings: hice reuses the he surrogate; dhice reuses the dice surrogate; all others use their own.

#### 3.2.2 Surrogate Inference

For each drive cycle $d$ present in the vehicle's age sequence, the surrogate is evaluated at the total laden mass at a representative age $a_d$ for that drive cycle:

$$\hat{x}_{p,d} = \beta_0^{(p,d)} + \sum_i \beta_i^{(p,d)}\, z_i + \sum_{i \leq j} \beta_{ij}^{(p,d)}\, z_i z_j, \qquad \mathbf{z} = \left(M^{\text{total}}_{k,p,a_d},\ \mu,\ P^{\text{acc}},\ \eta_p^{-1}\right)$$

The raw surrogate output is then mapped to per-fuel tank-side consumption $x_{f,a}$:

| Powertrain | Surrogate output | Mapping to $x_{f,a}$ |
|-----------|-----------------|----------------------|
| dice, he | diesel L/km | $x_{\text{diesel},a} = \hat{x}$ |
| be | kWh/km (battery) | $x_{\text{charge},a} = \hat{x}$ |
| fc | kg/km (H2) | $x_{\text{H2},a} = \hat{x}$ |
| hice | diesel L/km (via he surrogate) | $x_{\text{H2},a} = \hat{x} \cdot H_{\text{diesel}} / H_{\text{H2}}$ |
| dhice | diesel L/km total (via dice surrogate) | split by declared proportions (see below) |
| phe | separate CD and CS modes (see below) | -- |

**DHICE fuel split.** The surrogate gives total energy as a diesel-equivalent. This is partitioned into diesel and H2 fractions using LHV-weighted declared proportions $\phi_f$:

$$x_f = \frac{\hat{x} \cdot H_{\text{diesel}} \cdot \phi_f / H_f}{\phi_{\text{diesel}} + \phi_{\text{H2}}}$$

**PHE modes.** The PHE is evaluated separately in charge-depleting (CD) mode using the BEV surrogate and charge-sustaining (CS) mode using the HEV surrogate:

$$x^{\text{CD}}_{\text{charge},a} = \hat{x}^{\text{be}}_{d_a}\!\left(\eta_{\text{motor}}\right), \qquad x^{\text{CS}}_{\text{diesel},a} = \hat{x}^{\text{he}}_{d_a}\!\left(\eta_{\text{ice}}\right)$$

Annual fuel quantities for the PHE are computed in Section 3.4 from these per-km values and the electric range available at each age.

#### 3.2.3 Refuelling Efficiency Adjustment

The surrogate is trained on vehicle energy demand in tank or battery units. Fuel costs and emissions are quoted per source unit (grid kWh at the meter, kg H2 at the pump). The gross-up from tank-side to source units is:

$$\tilde{x}_{f,a} = x_{f,a}\, /\, \eta^{\text{fill}}_f$$

For fuels where $\eta^{\text{fill}}_f = 1$ (diesel, H2) this has no effect.

---

### 3.3 Range

The effective vehicle range is the binding energy-storage constraint across all components. For each energy storage component $c$ associated with fuel $f_c$:

$$d^{\text{max}}_{c,a} = \frac{X_c \cdot u_c}{\tilde{x}_{f_c,a}}$$

$$d^{\text{max}}_a = \min_c\, d^{\text{max}}_{c,a}$$

The binding component also determines the refuelling rate $r$ used in Section 3.4. For batteries, $r$ is the fast-charger wall power multiplied by the fast-charging efficiency (kW delivered to the battery). For H2 tanks, $r$ is pump flow in kg/h.

---

### 3.4 Annual Distance

**Working-day conversion.** Trucks operate 5 days out of every 7, so the daily target exceeds a simple $1/365$ fraction of the annual target:

$$d^{\text{daily}}_{k,a} = \frac{d^{\text{tgt}}_{k,a}}{365} \cdot \frac{7}{5}$$

**Battery degradation.** Before the range check each year, the effective range is reduced by capacity fade:

$$d^{\text{max}}_a \leftarrow d^{\text{max}}_a \cdot \max\!\left(0,\ 1 - \xi \cdot a - \theta \cdot N^{\text{cyc}}_a\right)$$

where $N^{\text{cyc}}_a$ accumulates year-by-year as:

$$N^{\text{cyc}}_a \leftarrow N^{\text{cyc}}_{a-1} + \frac{d_{k,p,y,a-1} \cdot \tilde{x}_{\text{charge},a-1} \cdot \eta^{\text{fill}}_{\text{charge}}}{X_{\text{battery}}}$$

**Range check and time-budget refuelling.** If $d^{\text{daily}}_{k,a} \leq d^{\text{max}}_a$, the vehicle drives the full daily target. Otherwise, extra distance is achieved via a refuelling or recharging stop. The time that would have been spent driving the unmet shortfall is available for the stop. Solving simultaneously for stop duration and extra distance achievable within that time budget:

$$d^{\text{extra}}_a = \max\!\left(0,\ \frac{\left(\dfrac{d^{\text{daily}}_{k,a} - d^{\text{max}}_a}{v_{k,a}} - 0.25\right) \cdot v_{k,a} \cdot r}{\tilde{x}_{f,a} \cdot v_{k,a} + r}\right)$$

where $0.25$ h is a fixed per-stop overhead and $f$ is the fuel of the binding energy storage component. The achievable daily distance is $d^{\text{max}}_a + d^{\text{extra}}_a$.

**Annual distance** (converting back to calendar-year basis):

$$d_{k,p,y,a} = \left(d^{\text{max}}_a + d^{\text{extra}}_a\right) \cdot \frac{5}{7} \cdot 365$$

$$d^{\text{en}}_a = d^{\text{extra}}_a \cdot \frac{5}{7} \cdot 365$$

**PHE annual fuel.** At each age the vehicle drives in CD mode up to the available electric range, then switches to CS mode for the remainder:

$$d^{\text{CD}}_a = \min\!\left(d^{\text{daily}}_{k,a},\ d^{\text{max,elec}}_a\right), \qquad d^{\text{CS}}_a = \max\!\left(0,\ d^{\text{daily}}_{k,a} - d^{\text{CD}}_a\right)$$

$$Q^{\text{CD}}_{\text{charge},a} = d^{\text{CD}}_a \cdot \tfrac{5}{7} \cdot 365 \cdot \tilde{x}^{\text{CD}}_{\text{charge},a}, \qquad Q^{\text{CS}}_{\text{diesel},a} = d^{\text{CS}}_a \cdot \tfrac{5}{7} \cdot 365 \cdot \tilde{x}^{\text{CS}}_{\text{diesel},a}$$

**Annual fuel (non-PHE):**

$$Q_{f,a} = d_{k,p,y,a} \cdot \tilde{x}_{f,a}$$

**En-route fast charging split (BETs).** When a battery-electric truck extends range via an en-route DC fast charger, that portion of energy is re-billed at fast-charge rates because the same battery energy incurs higher grid losses than at a depot AC charger:

$$Q_{\text{slow},a} \leftarrow Q_{\text{slow},a} - d^{\text{en}}_a \cdot \tilde{x}_{\text{slow},a}$$

$$Q_{\text{fast},a} = d^{\text{en}}_a \cdot \tilde{x}_{\text{slow},a} \cdot \frac{\eta^{\text{fill}}_{\text{slow}}}{\eta^{\text{fill}}_{\text{fast}}}$$

---

### 3.5 Fuel Cell Stack Replacements

Cumulative operating hours are tracked age-by-age:

$$h_a = h_{a-1} + \frac{d_{k,p,y,a}}{v_{k,a}}$$

A replacement occurs when accumulated hours exceed the stack lifetime $L^{\text{fc}}$, after which the counter resets to zero:

$$r^{\text{fc}}_a = \begin{cases} 1 & h_a > L^{\text{fc}} \\ 0 & \text{otherwise} \end{cases}$$

Replacement events enter the embodied emission and cost calculations at their actual ages (Sections 3.6 and 3.8).

---

### 3.6 Emissions

**Embodied emissions.** Manufacturing emissions are attributed at age 0; fuel cell stack replacements add further embodied emissions at their actual replacement ages:

$$\Gamma^{\text{emb}}_{k,p,y,a} = \begin{cases} \left(M^{\text{frame}}_k + \zeta_k \cdot M^{\text{trailer}}_k\right) i^{\text{emb}}_{\text{frame}} + \displaystyle\sum_{\text{conv}} M^{\text{conv}}_c \, i^{\text{emb}}_c + \displaystyle\sum_{\text{ESS}} X_c \, i^{\text{emb}}_c & a = 0 \\[6pt] r^{\text{fc}}_a \cdot M^{\text{conv}}_{\text{fc}} \cdot i^{\text{emb}}_{\text{fc}} & a > 0 \end{cases}$$

The first sum covers converters and transmissions (factor in kgCO2e/kg); the second covers energy storage components (factor in kgCO2e per unit of capacity).

**Supply-chain and tailpipe emissions** (annual, per vehicle):

$$\Gamma^{\text{supply}}_{k,p,y,a} = \sum_f Q_{f,a} \cdot i^{\text{supply}}_f$$

$$\Gamma^{\text{use}}_{k,p,y,a} = \sum_f Q_{f,a} \cdot i^{\text{use}}_f$$

$i^{\text{use}}_f = 0$ for all ZEV fuels (electricity, H2); there is no tailpipe combustion.

---

### 3.7 Capital Cost

$$C^{\text{capital}}_{k,p,y} = C^{\text{frame}}_k + \sum_c C^{\text{comp}}_c(y)$$

Component costs at model year $y$ (time-varying where noted). Dollar amounts ($C$) are fixed prices per item; rates ($c$) are multiplied by the rated capacity or power:

| Component | Symbol form | Units |
|-----------|-------------|-------|
| ICE engine | $C^{\text{engine}}_{k,p,y}$ | \$ |
| Electric motor | $c^{\text{motor}}_{k,p,y} \cdot P^{\text{motor}}_{k,p}$ | \$/kW $\times$ kW |
| Battery pack | $c^{\text{battery}}_{k,p,y} \cdot X_{\text{battery}}$ | \$/kWh $\times$ kWh (declining) |
| H2 tank | $c^{\text{tank}}_{k,p,y} \cdot X_{\text{tank}}$ | \$/kg $\times$ kg H2 (declining) |
| Fuel cell stack | $c^{\text{fc,buy}}_{k,p,y} \cdot P^{\text{fc}}_{k,p}$ | \$/kW $\times$ kW (declining) |
| Diesel tank | $c^{\text{dtank}}_{k,p,y} \cdot X_{\text{tank}}$ | \$/L $\times$ L |
| Depot charger | $\psi_{k,p} \cdot C^{\text{charger}}_y$ | fraction $\times$ \$/unit (declining) |
| Transmission | $C^{\text{trans}}_{k,p}$ | \$ |
| After-treatment | $C^{\text{aft}}_{k,p}$ | \$ (diesel powertrains only) |

$\psi_{k,p}$ is the charger fraction per vehicle: 1 for day cab and straight BETs, 0.25 for PHE, 0 for sleeper.

---

### 3.8 Annual Cost and Revenue

All cost terms are age-arrays over $a \in A$. Annual cost at age $a$ in calendar year $t = y + a$:

| Cost component | Annual cost at age $a$ |
|----------------|------------------------|
| Capital | $C^{\text{capital}}_{k,p,y}$ at $a = 0$; zero for $a > 0$ |
| O\&M | $d_{k,p,y,a} \cdot c^{O\&M}_p$ |
| Fuel | $\displaystyle\sum_f Q_{f,a} \cdot c_f(y+a)$ |
| Driver | $d^{\text{tgt}}_{k,a} \cdot c^{\text{driver}}_k$ (paid on target distance, not actual) |
| FC replacement | $r^{\text{fc}}_a \cdot P^{\text{fc}}_{k,p} \cdot c^{\text{fc,rep}}_p(y+a)$ |
| Carbon tax | see Section 6.1 |
| LCFS | see Section 6.2 |
| ZEV mandate | $C^{\text{nonZEV}}_t$ (non-ZEV) or $-C^{\text{ZEV}}_t$ (ZEV) at $a = 0$ only; see Section 6.3 |

Annual revenue:

$$\text{Rev}_{k,p,y,a} = d_{k,p,y,a} \cdot \frac{M^{\text{payload}}_{k,p,a}}{1000} \cdot \varsigma_k$$

---

### 3.9 NPV and TCO

Survival-weighted discount factor:

$$\Delta_a = \frac{\chi_{k,a}}{(1 + \delta)^a}$$

Total cost of ownership (discounted sum over all ages and cost components):

$$\text{TCO}_{k,p,y} = \sum_{a \in A} \Delta_a \sum_j C^{(j)}_{k,p,y,a}$$

Net present value from the operator's perspective:

$$\text{NPV}_{k,p,y} = \sum_{a \in A} \Delta_a\, \text{Rev}_{k,p,y,a}\ -\ \text{TCO}_{k,p,y}$$

A higher NPV indicates a more profitable vehicle. NPV is the utility term driving market share allocation (Section 5).

</details>

---

<details>
<summary><strong>4. Fleet Dynamics</strong></summary>

### 4.1 Activity Requirement

Total fleet activity (tonne-km per year) grows at rate $\kappa$ from the base year. Each vehicle type carries a fixed share $\alpha_k$ of total activity:

$$\mathcal{A}_k(t) = \mathcal{A}^{\text{tot}}_0 \cdot \alpha_k \cdot (1 + \kappa)^{t - t_0}$$

---

### 4.2 Initial Stock

The pre-2025 fleet is modelled as diesel-only cohorts for model years $y \in Y,\, y < t_0$. Cohort sizes are calibrated so that cumulative surviving activity in $t_0$ equals $\mathcal{A}_k(t_0)$.

The denominator $\Omega_k$ is the survival- and growth-weighted lifetime activity per vehicle, computed from the oldest surviving vintage as a reference:

$$\Omega_k = \sum_{a \in A} \alpha_{k,\text{dice},y_{\text{ref}},a} \cdot \chi_{k,a} \cdot (1 + \kappa)^{-a}$$

where $\alpha_{k,p,y,a} = d_{k,p,y,a} \cdot M^{\text{payload}}_{k,p,a} / 1000$ is the per-vehicle activity at age $a$.

Each cohort is then sized proportionally to the activity target, with growth and survival adjustments for its vintage year:

$$N_{k,\text{dice},y,t_0} = \frac{\mathcal{A}_k(t_0) \cdot (1 + \kappa)^{y - t_0} \cdot \chi_{k,\,t_0 - y}}{\Omega_k}$$

---

### 4.3 Year-by-Year Simulation

For each $t \in T$:

**Step 1 -- Roll-over.** Surviving vehicles from $t-1$ advance one year. The conditional survival ratio is applied (marginal rate of survival from age $a-1$ to $a$) rather than the raw survival rate, so each cohort declines at the correct marginal rate:

$$N_{k,p,y,t} = N_{k,p,y,t-1} \cdot \frac{\chi_{k,\,t-y}}{\chi_{k,\,t-1-y}}$$

**Step 2 -- Build new vehicles.** A fresh cohort $(k, p, t)$ is instantiated for every $(k, p)$ pair, with all parameters time-sliced to model year $t$.

**Step 3 -- Activity gap and new sales.** Activity already met by surviving cohorts:

$$\hat{\mathcal{A}}_k(t) = \sum_{p} \sum_{y < t} N_{k,p,y,t} \cdot \alpha_{k,p,y,\,t-y}$$

Weighted-average activity per new vehicle:

$$\bar{\alpha}_k(t) = \sum_p S_{k,p,t} \cdot \alpha_{k,p,t,0}$$

New sales allocated by market share:

$$N_{k,p,t,t} = S_{k,p,t} \cdot \frac{\max\!\left(\mathcal{A}_k(t) - \hat{\mathcal{A}}_k(t),\ 0\right)}{\bar{\alpha}_k(t)}$$

</details>

---

<details>
<summary><strong>5. Market Share</strong></summary>

### 5.1 Nested Logit

New vehicle purchases are allocated across powertrains using a McFadden nested logit, not a flat multinomial logit: powertrains are grouped into nests that share a family resemblance, so a shift between two similar options (e.g. diesel vs. mild hybrid) doesn't over-count relative to a shift toward a genuinely distinct alternative (e.g. battery electric). The nesting tree is fixed model structure, not a fitted or uncertain parameter:

```
Liquid (l = 0.7)
  Conventional (l = 0.4): dice, he
  phe
Hydrogen (l = 0.6): fc, hice, dhice
Electric (l = 1.0): be
```

`dhice` (a 75%-diesel/25%-H2 dual-fuel ICE) sits in Hydrogen, not Liquid -- grouped by ZEV-adjacent substitution pattern rather than fuel share.

**Leaf utility** of powertrain $p$ (before any production-cap shadow cost, Section 5.3):

$$V_p = \lambda \left(\text{NPV}_{k,p,t} - \mu_p\right)$$

**Inclusive utility**, bottom-up from leaves to root. For nest $n$ with children $c \in n$ and scale $\ell_n$:

$$U_n = \ell_n \, \ln\!\left(\sum_{c \in n} \exp\!\left(U_c \,/\, \ell_n\right)\right)$$

taking $U_c = V_c$ at leaves. The root's own scale is fixed at 1 (nothing sits above it to rescale against).

**Conditional share**, top-down from root to leaves. Within nest $n$, the probability of choosing child $c$ given $n$ is chosen:

$$P(c \mid n) = \frac{\exp\!\left(U_c \,/\, \ell_n\right)}{\displaystyle\sum_{c' \in n} \exp\!\left(U_{c'} \,/\, \ell_n\right)}$$

and $S_{k,p,t}$ is the product of $P(c\mid n)$ down the path from root to leaf $p$.

$\lambda$ controls overall price sensitivity; $\ell_n \to 0$ makes nest $n$'s members near-perfect substitutes (a shift mostly reallocates share *within* the nest), while $\ell_n = 1$ for every nest collapses the whole tree to exactly the flat multinomial logit, at any depth.

---

### 5.2 Production Cap

Nascent technologies cannot grow faster than their supply chain allows. The cap switches between two CAGR regimes at a threshold share $S^*$:

$$\bar{S}_{k,p,t} = \begin{cases} \dfrac{S_{k,p,t-1} \cdot \left(1 + z^{\text{nac}}_{k,p}\right)}{1 + \kappa} & S_{k,p,t-1} < S^* \\[8pt] \dfrac{S_{k,p,t-1} \cdot \left(1 + z^{\text{mat}}_{k,p}\right)}{1 + \kappa} & S_{k,p,t-1} \geq S^* \end{cases}$$

The denominator $(1 + \kappa)$ normalises for fleet growth so the cap governs share rather than absolute sales volume. A floor is applied: if $S_{k,p,t-1} < S^{\text{init}}_{k,p}\,/\,(1 + z^{\text{nac}}_{k,p})$, it is replaced by that floor before computing the cap.

---

### 5.3 Shadow Pricing

Production caps are enforced by solving for a shadow cost $\mu_p \geq 0$ per powertrain such that the resulting nested-logit share never exceeds its cap, with complementary slackness (a cap is only "paid for" once it actually binds):

$$\mu_p \geq 0, \qquad S_{k,p,t}(\boldsymbol{\mu}) \leq \bar{S}_{k,p,t}, \qquad \mu_p \left(\bar{S}_{k,p,t} - S_{k,p,t}(\boldsymbol{\mu})\right) = 0$$

A capped powertrain stays in the choice set at a discounted utility ($V_p$ falls as $\mu_p$ rises) rather than being removed and having its excess demand redistributed among the survivors -- removal would leak an inflated inclusive value up through its nest and distort every sibling nest's share, which is precisely the failure mode nesting is meant to avoid.

**Solved by Gauss-Seidel sweeps with per-powertrain bisection:** one powertrain's $\mu_p$ at a time, holding every other powertrain's $\mu$ fixed, cycling through all powertrains repeatedly until every capped powertrain's share is tight to its cap and every uncapped powertrain's share is feasible, to a relative tolerance of $10^{-5}$. Bisection (rather than a joint Newton step) is used because $S_{k,p,t}(\mu_p)$, holding everything else fixed, is monotonically non-increasing in $\mu_p$ -- bisection cannot overshoot regardless of how saturated a share is. A cap of exactly 0 permanently excludes that powertrain from the choice set for the call; a single remaining powertrain gets share 1 mechanically, with no cap check needed. $\mu_p$ is warm-started from its last-converged value across years and across the ZEV-mandate's repeated calls within a year (Section 6.3), which changes nothing about the converged answer, only how quickly it's reached.

</details>

---

<details>
<summary><strong>6. Policy Instruments</strong></summary>

Two application points exist in the vehicle construction step. Physics policies (GVWL exemption) modify vehicle parameters before the cohort is instantiated, so changes propagate through mass, fuel consumption, range, and cost. Cost policies (carbon tax, LCFS) write additional cost terms after instantiation and trigger a TCO/NPV recalculation. The ZEV mandate is endogenous and operates via an outer convergence loop around Steps 2 and 3 of the annual simulation.

### 6.1 Carbon Tax

Annual cost added to each vehicle cohort at operating age $a$ (calendar year $t = y + a$):

$$C^{\text{ct}}_{k,p,y,a} = \frac{\Gamma^{\text{supply}}_{k,p,y,a} + \Gamma^{\text{use}}_{k,p,y,a}}{1000} \cdot c^{\text{ct}}_t$$

$c^{\text{ct}}_t$ is linearly interpolated between anchor years and set to zero before $t_0$.

---

### 6.2 Low Carbon Fuel Standard (LCFS)

The LCFS penalises fuels with carbon intensity (CI) above an annually tightening, energy-basis standard and credits fuels below it. Unlike the fuel-and-vehicle-agnostic carbon tax, the LCFS standard is expressed per joule of energy delivered and adjusted per fuel by an Energy Effectiveness Ratio, crediting more-efficient energy carriers (e.g. electricity) even at equal combustion CI.

**Target carbon intensity per joule** in year $t$, anchored to a fixed reference diesel intensity $\text{CI}^{\text{diesel}}$ and diesel's own LHV $H_{\text{diesel}}$:

$$\text{TCI}(t) = \frac{\text{CI}^{\text{diesel}}}{H_{\text{diesel}}} \cdot \left(1 - T^{\text{lcfs}}_t\right)$$

$T^{\text{lcfs}}_t$ is the CI reduction target, linearly interpolated from $T^{\text{lcfs}}_{t_0} = 0.183$ to $T^{\text{lcfs}}_{t_{\text{end}}} = 0.76$ and held at zero before $t_0$.

**Annual LCFS cost**, summed over every fuel $f$ the vehicle consumes (negative values are credits):

$$C^{\text{lcfs}}_{k,p,y,a} = \frac{c^{\text{lcfs}}}{1000} \sum_f Q_{f,a} \cdot \left(i^{\text{supply}}_f + i^{\text{use}}_f - \text{TCI}(t) \cdot \text{EER}_{f,p} \cdot H_f\right)$$

$\text{EER}_{f,p} = 1$ for diesel fuel regardless of powertrain; $\text{EER}_{f,p} = \text{EER}_p$ (powertrain-specific, e.g. higher for electricity/H2 than for liquid fuels) for every other fuel. A powertrain with no entry in the EER table defaults to 1 for all its fuels (diesel-equivalent treatment).

---

### 6.3 ZEV Mandate

The mandate requires ZEV powertrains (battery electric, fuel cell, hydrogen ICE hybrid) to reach a target fraction $Z^{\text{tgt}}_t$ of new sales each year. It is enforced endogenously via a credit market: each vehicle sold is worth $\rho_k$ credits, non-ZEVs owe their share of the compliance obligation, and ZEVs are paid for the credits they generate -- applied at $a = 0$ to new vehicles in year $t$.

**Credit price.** A smooth logistic function of the compliance gap between the trial ZEV share $p^{\text{zev}}$ and the target -- near $c^{\text{penalty,max}}$ when deep below target, collapsing toward 0 at or above it, with no hard cliff:

$$c^{\text{credit}}(Z^{\text{tgt}}_t,\, p^{\text{zev}}) = \frac{c^{\text{penalty,max}}}{1 + \exp\!\left(k_\ell \left(p^{\text{zev}} - Z^{\text{tgt}}_t\right)\right)}, \qquad k_\ell = \frac{\ln 19}{w}$$

**Credit costs**, revenue-bounded so the government never pays out more than it collects. Non-ZEVs always owe their own flat share of the obligation:

$$C^{\text{nonZEV}}_t = \rho_k \cdot c^{\text{credit}}_t \cdot Z^{\text{tgt}}_t$$

The pool this raises is $C^{\text{nonZEV}}_t \cdot N^{\text{nonZEV}}$. ZEVs are paid the same flat market rate for their own credits if the pool covers it, otherwise payouts ration down proportionally so total payout never exceeds the pool:

$$\pi_t = \min\!\left(1,\ Z^{\text{tgt}}_t \cdot \frac{1 - p^{\text{zev}}}{\max(p^{\text{zev}},\,\epsilon)}\right), \qquad C^{\text{ZEV}}_t = \rho_k \cdot c^{\text{credit}}_t \cdot \pi_t$$

Non-ZEV new vehicles incur $+C^{\text{nonZEV}}_t$ at $a=0$; ZEV new vehicles receive $-C^{\text{ZEV}}_t$ at $a=0$. Net revenue is $\geq 0$ at the fixed point (within bisection tolerance): positive when ZEVs are undersupplied relative to target, $\approx 0$ once ZEV supply is abundant enough to exhaust the pool.

**Convergence loop.** $c^{\text{credit}}(Z^{\text{tgt}}_t, \cdot)$ is monotonically decreasing in $p^{\text{zev}}$, and the market's realised ZEV share responds monotonically non-decreasingly to a higher credit price, so their composition minus $p^{\text{zev}}$ is monotonic with at most one root. Solved by **bisection on $p^{\text{zev}} \in [0, 1]$** each year (no cross-year warm start -- each year starts fresh from $[0,1]$, matching how the reference Paper 1 model was run): probe the bracket midpoint, apply the implied credit price, run the nested-logit market-share step (Section 5) to measure the resulting ZEV share, narrow the bracket by comparison, repeat until the bracket width is below $10^{-4}$ or 30 iterations elapse. This is robust regardless of how steep the credit-price transition is, since it only ever uses the sign of the compliance gap, not its magnitude. Production-cap-bound years converge the same way, to whatever $p^{\text{zev}} < Z^{\text{tgt}}_t$ the market saturates at -- not a special case.

---

### 6.4 GVWL Exemption

ZEV powertrains receive additional GVWL headroom $\Delta M^{\text{GVWL}}_k$ before the vehicle cohort is constructed. This propagates through mass (Section 3.1), fuel consumption (Section 3.2), range (Section 3.3), and ultimately cost and NPV.

| Vehicle type | $\Delta M^{\text{GVWL}}_k$ (kg) |
|-------------|--------------------------------|
| sleeper | 5 000 |
| day\_cab | 3 000 |
| straight | 2 000 |

</details>

---

<details>
<summary><strong>7. Monte Carlo Framework</strong></summary>

### 7.1 Parameter Distributions

Each uncertain parameter is assigned a distribution. Given a uniform draw $u \in [0, 1]$, the realised value is:

| Distribution | Realisation |
|-------------|-------------|
| Constant $v$ | $\theta = v$ |
| Uniform$(a, b)$ | $\theta = a + (b - a)\,u$ |
| Triangular$(a, m, b)$ | $\theta = F^{-1}_{\text{triang}}(u;\,a,\,m,\,b)$ |
| Linear interp | Start and end values are themselves distributions; $\theta(Y)$ is a time series linearly interpolated from the realised start and end values |
| Piecewise interp | Anchor-year values are themselves distributions; $\theta(Y) = \mathrm{interp}(Y,\,\text{anchors},\,\text{realised values})$ |

---

### 7.2 Grouped Sampling

Parameters sharing a group label $\mathcal{G}$ receive a single common draw $u_{\mathcal{G}}$, preserving physical correlations across powertrains. For example, all embodied emission factors share one draw so that a decarbonised manufacturing scenario is applied consistently across all components and vehicle types. Parameters without a group label receive independent draws.

A single $(N^{\text{runs}} \times N^{\text{cols}})$ sample matrix is generated once from a fixed seed and shared across all scenarios. Because every scenario uses the same underlying draws, differences in outputs between scenarios are attributable solely to the policy instruments rather than sampling variation.

---

### 7.3 Convergence

After every $\Delta n$ completed runs (minimum 200), the accumulated results are split into two equal halves and the two-sample Kolmogorov--Smirnov statistic is computed for each monitored output series at each calendar year:

$$D_{\text{KS}} = \max_x \left|F_A(x) - F_B(x)\right|$$

$D_{\text{KS}} \in [0, 1]$ is the maximum vertical gap between the empirical CDFs of the two halves. It is scale-free and requires no normalisation. For $n$ i.i.d. draws per half, $\mathbb{E}[D_{\text{KS}}] \approx 0.8 / \sqrt{n}$, so the convergence rate is predictable.

Convergence is declared when $D_{\text{KS}} < \tau_{\text{KS}}$ for all monitored series and all years. Remaining queued runs are cancelled on convergence.

**Monitored series** (5 per vehicle type $k$, 15 total): aggregate ZEV stock, supply-chain fleet emissions, tailpipe fleet emissions, fleet capital costs, fleet fuel costs.

---

### 7.4 Scenarios

| Scenario | Active policies |
|----------|----------------|
| baseline | none |
| carbon\_tax | Carbon tax |
| lcfs | LCFS, model-derived EER set |
| lcfs\_bc\_eer | LCFS, BC LCFS technical regulation's legislated EER set (vs. model-derived) |
| zev\_mandate | ZEV mandate |
| gvwl | GVWL exemption |
| full\_policy | Carbon tax + LCFS (model-derived EER) + ZEV mandate + GVWL exemption |

</details>

---

<details>
<summary><strong>8. Outputs</strong></summary>

The model produces fleet-level aggregates for each calendar year $t \in T$ by summing cohort-level quantities over all surviving vehicles, weighted by cohort size.

**Total on-road stock** by powertrain:

$$\mathcal{N}_{k,p,t} = \sum_{y \in Y} N_{k,p,y,t}$$

**New sales** of powertrain $p$, type $k$, in year $t$: $N_{k,p,t,t}$ (defined in Section 4.3).

**Aggregate ZEV stock** for type $k$:

$$\mathcal{N}^{\text{zev}}_{k,t} = \sum_{p \in P^{\text{zev}}} \mathcal{N}_{k,p,t}, \qquad P^{\text{zev}} = \{\text{be, fc, hice}\}$$

**Fleet emissions** (kgCO2e/yr) for each stream $(\cdot) \in \{\text{emb, supply, use}\}$:

$$\mathcal{G}^{(\cdot)}_{k,t} = \sum_p \sum_{y} N_{k,p,y,t} \cdot \Gamma^{(\cdot)}_{k,p,y,\,t-y}$$

**Fleet fuel consumption** (source units/yr):

$$\mathcal{Q}_{k,f,t} = \sum_p \sum_{y} N_{k,p,y,t} \cdot Q_{f,\,t-y}$$

**Fleet system costs** (\$/yr). For flow cost categories (O\&M, fuel, driver, FC replacements, carbon tax, LCFS, ZEV mandate):

$$\mathcal{C}^{(j)}_{k,t} = \sum_p \sum_{y} N_{k,p,y,t} \cdot C^{(j)}_{k,p,y,\,t-y}$$

Capital costs are attributed at the point of sale: capital cost in year $t$ is $\sum_p N_{k,p,t,t} \cdot C^{\text{capital}}_{k,p,t}$.

Two further series are recorded directly, not aggregated: $\text{NPV}_{k,p,y}$ (Section 3.9) itself, for $y \in \{2030, 2040, 2050\}$, as a per-run scalar rather than an annual time series; and, for scenarios with an active ZEV mandate, the realised credit price relative to its ceiling, $c^{\text{credit}}_t \,/\, c^{\text{penalty,max}}$, per calendar year.

Each aggregate is computed independently for every Monte Carlo run, producing a distribution over outcomes that reflects the full joint uncertainty across all uncertain parameters.

</details>
