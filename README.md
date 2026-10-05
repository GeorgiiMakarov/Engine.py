# Weil-Form Computation Engine — component-wise rigor status

Not called "Certified Weil-Form Computation": as of this revision only
ONE of the three error-budget components is actually derived. Calling the
whole thing "certified" would be a documentation claim unsupported by the
code. Rename it once all three components in `ErrorBudget` are real.

## Interface (as requested)

```python
from engine import ZETA, CHI3, weil_form, galerkin_matrix, certify_positive, certify_negative, C

C(ZETA, N=4, c=50)
# -> {"verdict": ..., "reason": ..., "budget_status": "PARTIAL (numerical component only)", ...}
```

## Error budget status (`ErrorBudget` in engine.py)

| Component | Status | What it would take |
| --- | --- | --- |
| `numerical` | **Derived.** Backward-error bound for LDL^T from the tracked elimination growth factor (Higham-style). | Done. |
| `archimedean` | Not derived. Placeholder `0`. | Bound the quadrature tail beyond `R=sqrt(80/a)` + `mp.quad`'s own discretisation error. Real work, not a quick add. |
| `prime` | Not derived. Placeholder `0`. | This is not a software task — it's the open question this whole conversation has been circling (de Branges completeness radius / Conrey–Li obstruction territory for this specific construction). Treat as research, not a TODO comment. |

`C(...)["budget_status"]` says `PARTIAL` for exactly this reason — with two
of three components at `0`, a "positive" verdict is a genuine lower bound
on rigor achieved so far, not a certificate. Do not let a clean dataclass
make this look more finished than it is.

## Validation of the independently derived explicit formula

`validate_explicit_formula.py` derives the classical Weil explicit formula
(archimedean digamma term + pole term + prime/von Mangoldt term) from
scratch and checks it against the first 10 known Riemann zeta zeros
(standard tabulated values). Agreement: **2.5e-19 relative error** at
moderate test-function width. This is a correctness check on the kernel
implementation, not a result — everything else is built on it, but it
belongs here, not in a headline.

## What is NOT validated — read before trusting any output

1. **Basis conditioning.** The Galerkin matrix uses a Gaussian test-function
family (h_a(r)=e^{-ar²}), chosen because Gaussians are closed under
multiplication, which lets every matrix entry reuse the single validated
formula. This is elegant but numerically **ill-conditioned**: by N=4–6
the smallest pivot is swamped by conditioning noise, not signal — and
the growth factor tracked in `ldlt()` now makes this visible directly
in the numerical budget, instead of hiding behind a flat constant.
2. **The χ₃ branch is unvalidated.** `CHI3` in `engine.py` is my from-scratch
generalization to Dirichlet L(s,χ₃) (odd character, no pole). I have
**not** checked it against known L(s,χ₃) zero locations the way ζ was
checked. `demo.py` shows a consistent, non-noise-level negative pivot
(−1.42) for χ₃ — interesting if real, equally consistent with a sign
bug in the odd-character archimedean shift. **Do not cite this number
anywhere until validated.**

## Honest next steps, in priority order

1. Validate CHI3 the same way ZETA was validated (need accurate L(s,χ₃)
zero locations — not from memory, pull from LMFDB).
2. Replace the Gaussian basis with an orthogonalized version (Gram–Schmidt
or SVD-whitened) or the actual CvS sin-kernel basis, to separate real
positivity signal from conditioning noise.
3. Implement a real archimedean + prime-cutoff error budget, not just a
precision-margin proxy.
4. Only after 1–3: treat verdicts as meaningful enough to build on.

## Reproduce

```bash
pip install mpmath sympy
python validate_explicit_formula.py   # explicit-formula check vs 10 known zeta zeros
python demo.py                        # Galerkin positivity grid (zeta + chi_3)
```
