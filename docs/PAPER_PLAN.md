# Paper plan — a parametric, well-conditioned drift-diffusion PINN

**Status:** working plan. This is a research plan, not a claim of results. It is
deliberately honest about what is novel, what is reproduction, and where the
project can fail.

---

## 1. One-sentence thesis

*A single physics-informed neural network, trained once under a principled
conditioning recipe, can act as a **parametric surrogate** for a semiconductor
drift-diffusion device — producing the full bias- (and doping-) dependent device
state at inference cost — and we validate it rigorously against a matched
in-house finite-element solver.*

## 2. Why this framing (and not "a PINN solves a diode")

A 1-D PN-diode drift-diffusion PINN, on its own, is **reproduction, not
research** — it has been done. To be publishable the paper must carry a genuine
contribution. This plan stacks three, so the paper stands even if the most
ambitious one underperforms:

1. **A conditioning recipe + ablation** (methods contribution). Drift-diffusion
   PINNs are notoriously hard because De Mari scaling makes the net doping
   `C = N/nᵢ ~ 1e6` and the depletion layer sub-1% of the domain. We give a
   recipe — hard boundary constraints with a depletion-matched potential
   baseline, normalised input, doping-scaled residual normalisation, L-BFGS
   polishing — and a **systematic ablation** quantifying each ingredient's
   effect on convergence, accuracy, and training stability across seeds. Ablation
   studies are publishable when systematic and reproducible.
2. **A parametric surrogate** (capability contribution). One network over
   `(x, V_applied, [N_A, N_D])` gives amortised I–V and doping design sweeps from
   a single training run — a capability the FEM solver structurally lacks.
3. **Validation against a matched FEM oracle** (rigor contribution). Most
   DD-PINN papers validate against a black-box commercial TCAD tool or analytics
   only. Owning `DDM.SPC` — an FEM solver in the *same* scaled variables — lets
   us compare **per-node fields**, not just terminal observables, which is a
   stronger validation than is common in this literature.

**Honest caveat:** parametric PINNs and DD-PINNs both exist. The novelty is the
*combination* — the conditioning recipe that makes the parametric DD surrogate
actually converge and its per-node validation — not any single element. The
paper must cite and benchmark against prior DD-PINN work to substantiate this.

## 3. Scope

- **In scope (core):** 1-D PN-junction diode, silicon + GaAs, equilibrium and
  forward bias, recombination-free ideal-diffusion first then optional SRH.
- **Stretch:** inverse parameter extraction from synthetic I–V; a 2-D diode or
  an added recombination channel.
- **Explicitly out of scope (this paper):** full 2-D MOSFET/HEMT, Schrödinger–
  Poisson quantum coupling — these are the *next* papers, kept out so this one
  stays finishable.

## 4. Method (what we write up)

- **Governing equations:** the coupled Poisson + electron/hole continuity system
  in De Mari scaled quasi-Fermi variables `(u,v,w)` (already implemented).
- **Trial solution:** neural field with the hard-BC output transform and the
  smooth depletion-matched potential baseline (§ derivation of the baseline
  curvature bound `~V_bi/W_dep²` is a clean, citable sub-result).
- **Loss:** strong-form residual MSE, doping-normalised; boundary term made
  redundant by the hard constraint.
- **Parametric extension:** append the bias (and doping) parameters to the
  network input; sample collocation points jointly over space and parameter
  space; a single training pass amortises the whole family.
- **Current functional:** replace the naive mean-flux current with a
  **flux-based terminal-current functional** (needed for quantitative I–V — see
  Risk R1).
- **Optimiser:** Adam warm-up + L-BFGS, with (new) **adaptive loss weighting**
  (NTK / gradient-norm balancing) for the forward-bias regime.

## 5. Experiments and figures

| # | Experiment | Delivers | Figure / Table |
|---|------------|----------|----------------|
| E1 | Equilibrium fields vs FEM (Si) | per-node L2/L∞ of u,n,p; V_bi; mass action | Fig 2 |
| E2 | **Conditioning ablation** (drop each ingredient; N seeds) | final residual, field error, success rate, train stability | Fig 3, **Table 1** |
| E3 | Forward I–V + ideality (Si) vs FEM | I–V (lin+semilog), ideality factor, current-conservation spread | Fig 4 |
| E4 | **Parametric surrogate** over bias (then + doping) | amortised I–V family, design-space heatmap, train-once-vs-N-solves speedup | Fig 5, **Table 3** |
| E5 | Robustness: materials (Si/GaAs), asymmetric doping, lengths | accuracy table across devices | **Table 2** |
| E6* | Inverse extraction from synthetic noisy I–V | recovered τ, μ; bias/variance | Fig 6 |
| E7* | 2-D diode *or* SRH recombination | extensibility demonstration | Fig 7 |

(* = stretch.)

- **Fig 1:** device schematic + network/hard-BC/baseline architecture diagram.

## 6. Rigor checklist (what reviewers will demand)

- **Baselines:** (a) a *vanilla* PINN (soft BC, no conditioning) to show the
  recipe matters; (b) at least one *prior DD-PINN method* from the literature,
  reimplemented, to show we are competitive/better. Without (b) the methods
  claim is weak.
- **Statistics:** ≥ 5 random seeds; report mean ± std and **success rate**
  (fraction of seeds reaching a residual/accuracy threshold). PINN papers that
  show one lucky run are correctly distrusted.
- **Metrics, defined precisely:** relative L2 and L∞ of each field vs FEM;
  ideality-factor error; I–V relative error; current-conservation spread.
- **Compute:** wall-clock train + inference vs FEM solve cost, hardware stated;
  the surrogate's value is *amortised* cost over a sweep, so report break-even N.
- **Convergence study:** error vs collocation count and vs network size.
- **Reproducibility:** seeds, configs, and the released repo (already public).

## 7. Related work (to complete with a real literature search)

Areas to survey and position against — representative, **to be verified and
cited properly** (do not cite from memory):

- Drift-diffusion / van Roosbroeck PINNs for devices (diodes, MOSFETs).
- Poisson–Nernst–Planck PINNs (same stiff structure, electrochemistry).
- Hard-constraint / boundary-encoding PINNs (output transforms, exact BCs).
- Adaptive loss weighting (learning-rate annealing, NTK, gradient-norm).
- Parametric / operator-learning surrogates (parametric PINNs, DeepONet, FNO)
  for PDE families.
- Inverse problems / differentiable physics for parameter identification.

**Action item:** a dedicated literature search is a prerequisite before writing
§related work and before finalising the novelty claim; if a very close
parametric DD-PINN already exists, we pivot the emphasis (e.g., toward the
matched-oracle validation or the inverse problem).

## 8. Target venues

- **Early feedback:** a workshop — NeurIPS/ICML *Machine Learning for Physical
  Sciences (ML4PS)* or *ML for Science* — to get the method in front of the
  community fast.
- **Primary full paper:** *Journal of Computational Electronics* (best fit —
  receptive to TCAD + ML) or *Solid-State Electronics*.
- **Higher bar / if device results are strong:** *IEEE T-ED*.
- **If methods-heavy:** *Journal of Computational Physics*.
- **Software route (separate):** *JOSS*, positioning `DDM.SPC` + `DDM_PINN` as a
  tool — but only once the capability set and test coverage are mature.

## 9. Milestones (effort-ordered; M1 is a go/no-go gate)

- **M1 — Quantitative I–V (GATE).** Flux-based current functional + adaptive
  weighting; reproduce the ideal-diode law with ideality ≈ 1 and I–V within a
  few % of FEM. *If this cannot be reached, the parametric-surrogate paper is not
  viable and we fall back to a methods/conditioning-only paper on equilibrium +
  parametric fields (weaker, but honest).*
- **M2 — Ablation harness.** Seed sweeps, metric logging, config system →
  Table 1, Fig 3.
- **M3 — Parametric surrogate.** Bias first, then doping → Fig 5, Table 3.
- **M4 — Robustness.** Si/GaAs, asymmetric doping, lengths → Table 2.
- **M5 — Writing + stretch.** Draft; inverse extraction (E6) if time allows.

## 10. Risks and mitigations

- **R1 — Forward I–V accuracy (highest risk).** The terminal current is a small
  net flux from exponentially sensitive minority injection. *Mitigation:*
  flux-based functional, adaptive weighting, minority-carrier-focused sampling.
  *Fallback:* methods/conditioning paper without the tight-I–V claim.
- **R2 — Parametric accuracy over wide ranges.** May degrade far from training
  density. *Mitigation:* bound the parameter ranges, report accuracy vs.
  distance from training support honestly.
- **R3 — Insufficient novelty.** *Mitigation:* the ablation rigor + matched-oracle
  per-node validation + parametric amortisation together; strengthened by an
  honest benchmark against a prior method (rigor checklist item).
- **R4 — Compute (CPU-bound here).** Parameter-space sampling multiplies cost.
  *Mitigation:* modest ranges for the paper; note GPU scaling.

## 11. Immediate next actions

1. Implement the flux-based current functional + adaptive loss weighting (M1).
2. Validate ideality factor vs FEM on the Si diode.
3. Only then build the ablation harness and the parametric surrogate.

---

## 12. Literature landscape (searched Aug 2026) — and a strategic pivot

A literature search changed the recommendation. **Both original headline
contributions are substantially occupied by 2024–2025 work.** The honest reading:

### What is already done (must cite; some must benchmark)

- **DDNet** (Riganti, Alasio, Bellotti, Dal Negro, arXiv:2509.08073, Sep 2025) —
  a *unified physics-informed DD solver, forward and inverse, mesh-free*, using
  three subnetworks (φ-Net, n-Net, p-Net) learning ψ, n, p directly, validated
  against traditional simulators. **This is the closest prior work and occupies
  the "forward+inverse mesh-free DD-PINN" and much of the "inverse" ground.**
  Must cite *and* benchmark against.
- **Out-of-training-range TCAD PINN / PI-DeepONet** (arXiv:2408.07921, 2024) —
  bias *generalization / extrapolation* of device fields (Si nanowire, MAPE
  ~0.1–0.2%). **Occupies the "parametric / bias-generalization surrogate"
  ground.** Must cite.
- **PI-DeepONet for drift-diffusion** (ICML 2025, arXiv:2505.04263) — operator
  learning + parameter identification for DD (on metric graphs). Parametric +
  inverse via operator learning.
- **IEEE 2023** "A PINN Algorithm for Simulating Semiconductor Devices"
  (IEEE Xplore 10249274) — early DD-PINN forward solver.
- Classical quantum-corrected DD (background, not ML): Schrödinger–Poisson-DD
  (SPDD), density-gradient, Bohm-potential (QDD) — established FEM methods.

### What appears open (the differentiated niche)

- **A self-consistent Schrödinger–Poisson-*Drift-Diffusion* PINN** — i.e.,
  quantum-**confinement-corrected transport** solved mesh-free as a physics
  residual — was **not found** in the search. The nearest neighbour is
  **Singh 2025** (*Real-Time Electrostatics of Double-Gate FETs with Neural
  Networks*, Wiley Adv. Theory Simul.), but that is a **data-driven DNN
  surrogate** for quantum-corrected **electrostatics only** (Poisson +
  Schrödinger, no DD transport). Our differentiators would be: (i) full
  quantum-corrected **transport** (terminal current), not just electrostatics;
  (ii) a **physics-residual PINN** (mesh-free, self-consistent), not a
  data-trained surrogate; (iii) autodiff self-consistency, potentially without
  the FEM solver's outer Gummel loop.

  *Caveat:* the arxiv/adsabs/semanticscholar hosts were egress-blocked, so this
  "open" conclusion rests on secondary sources and must be confirmed with a full
  scholar search before the novelty claim is finalised.

### Revised positioning

- **New headline contribution:** a self-consistent **Schrödinger–Poisson
  quantum-corrected drift-diffusion PINN**, validated per-node against `DDM.SPC`
  (which implements exactly this via its Gummel loop) — the one direction that is
  (a) apparently open, (b) uniquely enabled by the parent FEM solver's SP
  capability, and (c) hard for the crowded plain-DD-PINN work to have pre-empted.
- **Demoted to supporting contributions:** the quasi-Fermi De Mari **conditioning
  recipe + ablation** (a genuine methodological contrast to DDNet's
  direct-density three-subnetwork formulation — learning O(1) quasi-Fermi
  potentials instead of 12-decade densities), and **parametric amortisation**
  over the parameters that matter for confinement (gate bias, body thickness).
- **Unchanged:** the matched-FEM-oracle per-node validation (still a rigor
  advantage), and the **M1 I–V gate** (quantum-corrected transport still needs
  the forward-bias transport solved first, so M1 remains the prerequisite).

### Consequences for the plan

- The quasi-Fermi-vs-direct-density conditioning study becomes a concrete,
  defensible experiment *because* DDNet exists to contrast against.
- A DDNet reimplementation (or careful qualitative comparison) is now a rigor
  requirement, not optional.
- Sivang/Singh electrostatics surrogate must be positioned against explicitly.
- Phase order shifts: after M1 (I–V), prioritise the **quantum-corrected**
  extension over the pure parametric surrogate.
