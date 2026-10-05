"""
Weil-Form Computation Engine — minimal, independently-derived, validated core.

Implements the requested interface:
weil_form(L_function)
galerkin_matrix(L_function, N, c)
tail_certificate(Q)
certify_positive(Q) / certify_negative(Q)
C(L, N, c) -> {"positive", "negative", "inconclusive"}

IMPORTANT HONESTY NOTE (read before trusting numbers):
This is MY OWN from-scratch implementation of the classical Weil explicit
formula, independently derived and validated against the first 10 known
Riemann zeta zeros (see validate_explicit_formula.py — matches to 1e-19
relative error at moderate test-function width). It is NOT a byte-for-byte
reproduction of the Connes-van Suijlekom / Connes-Consani-Moscovici /
Groskin sin-kernel Galerkin recipe discussed earlier in this conversation —
I do not have their exact D_prime/D_pole normalization from primary source,
only fragments. The basis choice here (Gaussian test functions, closed
under multiplication) is a different, self-consistent finite-dimensional
realization of the same Weil-positivity idea, chosen specifically because
it lets every matrix entry reuse the SAME validated single-variable formula.
Treat this as a correct, working reference engine — not a claim of matching
anyone else's exact published digits.
"""

import mpmath as mp
from sympy import primerange
from dataclasses import dataclass

mp.mp.dps = 30


@dataclass
class ErrorBudget:
    """Component-wise error budget. Read the status of each field before
    trusting `total` — they are NOT comparable in rigor or effort."""
    archimedean: mp.mpf   # STATUS: not yet derived. Needs quadrature-tail
                          # bound beyond R=sqrt(80/a) + a real discretisation
                          # bound for mp.quad. Set to 0 = placeholder, not "no error".
    prime: mp.mpf          # STATUS: not yet derived, and NOT a quick add.
                          # This is the open research question (de Branges
                          # completeness radius / Conrey-Li obstruction
                          # territory) discussed at length in this session,
                          # not a software task. Set to 0 = placeholder.
    numerical: mp.mpf      # STATUS: derived below — backward-error bound
                          # for LDL^T from the observed growth factor.

    @property
    def total(self):
        return self.archimedean + self.prime + self.numerical

    @property
    def is_fully_rigorous(self):
        return self.archimedean > 0 and self.prime > 0


# --------- L-function descriptors ----------

class LFunction:
    """Minimal descriptor: archimedean parity + local coefficients."""
    def __init__(self, name, archimedean_shift, local_coeff, has_pole):
        self.name = name
        self.archimedean_shift = archimedean_shift  # 0 for zeta / even chi, 1 for odd chi
        self.local_coeff = local_coeff              # function p,k -> coefficient (e.g. chi(p^k))
        self.has_pole = has_pole                     # zeta: True; non-principal Dirichlet L: False


def weil_form(L: LFunction):
    """Returns a callable kernel(a) = pole(a) + archimedean(a) - 2*prime_sum(a, c)
    i.e. the explicit-formula RHS builder for this L-function, parameterised
    by Gaussian test-function width a and prime cutoff c."""

    def h(r, a):
        return mp.e ** (-a * r * r)

    def g(u, a):
        return 1 / (2 * mp.sqrt(mp.pi * a)) * mp.e ** (-u * u / (4 * a))

    def kernel(a, c):
        a = mp.mpf(a)
        pole = mp.mpf(0)
        if L.has_pole:
            pole = 2 * mp.e ** (a / 4)

        shift = L.archimedean_shift

        def integrand(r):
            return h(r, a) * mp.re(mp.digamma(mp.mpf('0.25') + shift / mp.mpf(2) + 1j * r / 2))

        R = mp.sqrt(80 / a)
        arch = (1 / (2 * mp.pi)) * mp.quad(integrand, [-R, 0, R])
        arch -= g(0, a) * mp.log(mp.pi)

        prime_sum = mp.mpf(0)
        for p in primerange(2, int(c) + 1):
            logp = mp.log(p)
            k = 1
            while True:
                coeff = L.local_coeff(p, k)
                if coeff != 0:
                    val = coeff * logp * mp.mpf(p) ** (-mp.mpf(k) / 2) * g(k * logp, a)
                    prime_sum += val
                if mp.mpf(p) ** (-mp.mpf(k) / 2) * logp < mp.mpf('1e-30') and k > 1:
                    break
                k += 1
                if k > 60:
                    break

        return pole + arch - 2 * prime_sum

    return kernel


# --------- Galerkin discretisation ----------

def galerkin_matrix(L: LFunction, N, c, a_min=0.008, a_step=0.006):
    """N x N symmetric matrix Q_ij = explicit-formula RHS applied to the
    product test function h_{a_i} * h_{a_j} = h_{a_i+a_j} (Gaussians are
    closed under multiplication, so this reuses the validated kernel exactly)."""
    kernel = weil_form(L)
    a_list = [mp.mpf(a_min) + i * mp.mpf(a_step) for i in range(N)]
    Q = mp.matrix(N, N)
    for i in range(N):
        for j in range(i, N):
            val = kernel(a_list[i] + a_list[j], c)
            Q[i, j] = val
            Q[j, i] = val
    return Q, a_list


# LDL^T certification

def ldlt(Q):
    """Manual symmetric LDL^T (no pivoting) — returns (pivots, growth_factor).
    growth_factor = max abs entry seen across all elimination steps, divided
    by max abs entry of the original Q. This is the standard quantity that
    backward-error bounds for unpivoted symmetric elimination are stated in
    terms of (Higham, ASNA, Ch.10) — NOT something GPT's recommendation
    mentioned, but it's what makes the numerical bound below honest rather
    than asserted."""
    n = Q.rows
    A = Q.copy()
    max_orig = max(abs(Q[i, j]) for i in range(n) for j in range(n))
    max_seen = max_orig
    pivots = []
    for k in range(n):
        d = A[k, k]
        pivots.append(d)
        if abs(d) < mp.mpf('1e-25'):
            pivots += [mp.mpf('nan')] * (n - k - 1)
            break
        for i in range(k + 1, n):
            factor = A[i, k] / d
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
                if abs(A[i, j]) > max_seen:
                    max_seen = abs(A[i, j])
    growth_factor = max_seen / max_orig if max_orig > 0 else mp.mpf(1)
    return pivots, growth_factor


def numerical_error_budget(Q, growth_factor, n):
    """Derived (not asserted) backward-error bound for unpivoted LDL^T:
    |computed LDL^T - Q|_max <~ c*n^3 * u * growth_factor * max|Q_ij|,
    standard form per Higham ASNA Thm 10.4-family results. c=10 is a
    deliberately conservative constant, not fitted."""
    u = mp.mpf(10) ** (-mp.mp.dps + 2)   # unit roundoff proxy for working precision
    max_q = max(abs(Q[i, j]) for i in range(Q.rows) for j in range(Q.rows))
    return 10 * n**3 * u * growth_factor * max_q


def tail_certificate(pivots, budget: ErrorBudget):
    if any(mp.isnan(p) for p in pivots):
        return "inconclusive", "near-singular pivot encountered"
    margin = budget.total
    status_note = "" if budget.is_fully_rigorous else \
        " [PARTIAL BUDGET: archimedean/prime not yet derived — margin is a floor, not a certificate]"
    min_pivot = min(pivots)
    if min_pivot > margin:
        return "positive", f"min pivot {mp.nstr(min_pivot,6)} > budget {mp.nstr(margin,6)}{status_note}"
    if min_pivot < -margin:
        return "negative", f"min pivot {mp.nstr(min_pivot,6)} < -budget {mp.nstr(margin,6)}{status_note}"
    return "inconclusive", f"min pivot {mp.nstr(min_pivot,6)} inside ±budget band{status_note}"


def certify_positive(Q, budget: ErrorBudget = None):
    pivots, gf = ldlt(Q)
    if budget is None:
        budget = ErrorBudget(archimedean=mp.mpf(0), prime=mp.mpf(0),
                             numerical=numerical_error_budget(Q, gf, Q.rows))
    verdict, reason = tail_certificate(pivots, budget)
    return verdict == "positive", reason


def certify_negative(Q, budget: ErrorBudget = None):
    pivots, gf = ldlt(Q)
    if budget is None:
        budget = ErrorBudget(archimedean=mp.mpf(0), prime=mp.mpf(0),
                             numerical=numerical_error_budget(Q, gf, Q.rows))
    verdict, reason = tail_certificate(pivots, budget)
    return verdict == "negative", reason


# --------- Top-level orchestrator ----------

def C(L: LFunction, N, c):
    Q, a_list = galerkin_matrix(L, N, c)
    pivots, growth_factor = ldlt(Q)
    budget = ErrorBudget(
        archimedean=mp.mpf(0),   # not yet derived — see ErrorBudget docstring
        prime=mp.mpf(0),         # not yet derived — see ErrorBudget docstring
        numerical=numerical_error_budget(Q, growth_factor, N),
    )
    verdict, reason = tail_certificate(pivots, budget)
    return {
        "verdict": verdict,
        "reason": reason,
        "N": N,
        "c": c,
        "growth_factor": mp.nstr(growth_factor, 4),
        "numerical_budget": mp.nstr(budget.numerical, 4),
        "budget_status": "PARTIAL (numerical component only — see ErrorBudget)",
        "min_pivot": mp.nstr(min(p for p in pivots if not mp.isnan(p)), 8) if not all(mp.isnan(p) for p in pivots) else "n/a",
    }


# --------- L-function library ----------

ZETA = LFunction(
    name="zeta",
    archimedean_shift=0,
    local_coeff=lambda p, k: 1,
    has_pole=True,
)


def dirichlet_chi3(p, k):
    # chi_3: nontrivial character mod 3 (chi(1)=1, chi(2)=-1, chi(0 mod 3)=0)
    r = p % 3
    if r == 0:
        return 0
    base = 1 if r == 1 else -1
    return base ** k


CHI3 = LFunction(
    name="L(s,chi_3)",
    archimedean_shift=1,   # chi_3 is odd
    local_coeff=dirichlet_chi3,
    has_pole=False,
)
