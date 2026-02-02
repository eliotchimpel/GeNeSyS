# GeNeSyS — Geometric Non-local System for Late-Time Cosmology

## Overview

**GeNeSyS** (Geometric Non-local System) is a research framework exploring whether a significant part of the cosmological dark sector can emerge from a **causal, non-instantaneous geometric response of spacetime**, rather than from additional fundamental particles or fluids.

The framework is deliberately formulated as a **late-time effective cosmological model**.  
It does **not** aim to replace ΛCDM, nor to describe primordial physics (inflation, recombination, or the origin of the CMB).  
Instead, it investigates whether dark-sector phenomenology can arise from geometry alone, within a controlled, testable, and explicitly bounded domain of validity.

All results are numerical, reproducible, and reported transparently, including controlled failures.

---

## Scientific Philosophy

GeNeSyS is built around a strict set of principles:

- No new fundamental matter fields (no particle dark matter, no dark energy fluid)
- No modification of local General Relativity
- No acausal dynamics
- No new fundamental constants
- Explicitly **post-primordial** (late-time cosmology only)
- Effective field theory mindset (no UV completion claimed)

The central hypothesis is minimal:

> When spacetime is described at cosmological scales, after coarse-graining local degrees of freedom, its geometric response may become **non-instantaneous** and depend on its own past history.

This temporal dependence is referred to as **geometric memory**.

This memory is:
- not a substance,
- not a field,
- not a particle,
- not tied to a specific microphysical sector.

It is a **property of the effective description**, analogous to viscosity or friction in macroscopic physics.

---

## Conceptual Roadmap (EB Pipeline)

The project follows a progressive validation pipeline:

| Stage | Purpose |
|------|--------|
| EB0 | Late-time background reconstruction (SN Ia + BAO) |
| EB01 | Passive auxiliary geometric response |
| EB02 | Activation of geometric memory |
| EB03 | Fully coupled emergent dynamics |
| EB03-LT | Late-time prediction without observational calibration |
| EB04 | Geometric projection to CMB scales (diagnostic only) |

Each stage:
- depends only on previous outputs,
- introduces no new observational input,
- is fully reproducible and auditable.

---

## Repository Structure

```
.
├── scripts/
│   ├── generate_v34_like_output.py
│   ├── generate_v34_multiscale_output.py
│   ├── generate_v34_bounded_output.py
│   ├── test_bridge_v34_to_v10.py
│   └── auxiliary utilities
│
├── data/
│   ├── sn_ia/
│   ├── bao/
│   └── fiducial_inputs/
│
├── outputs/
│   ├── run_v34_output.npz
│   ├── out_bridge_*/
│   └── logs/
│
├── docs/
│   ├── GeNeSyS_10_1_Model_Presentation.pdf
│   ├── GeNeSyS_10_1_Technical.pdf
│   ├── GeNeSyS_10_5_SNe_and_BAO.pdf
│   ├── GeNeSyS_10_6_Testing_Pipeline.pdf
│   ├── GeNeSyS_10_6_Perturbations_Closure.pdf
│   └── Public_Overview.pdf
│
├── README.md
└── LICENSE
```

---

## Installation

### Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

---

## Usage Guide

### Step 1 — Generate a bounded microscopic dataset

```bash
python3 scripts/generate_v34_bounded_output.py
```

### Step 2 — Test the micro-to-macro bridge

```bash
python3 scripts/test_bridge_v34_to_v10.py \
  --input_npz run_v34_output.npz \
  --outdir out_bridge_bounded \
  --kernel_len 60 \
  --ridge 0.1 \
  --plots
```

---

## Interpretation of Results

The diagnostics show that memory survives coarse-graining and requires a multi-timescale (bimodal) kernel, while local closures fail.

---

## License

Released under an open research license. See `LICENSE`.
