import time
from engine import C, ZETA, CHI3, galerkin_matrix, ldlt
import mpmath as mp

print("=== C(zeta, N, c) across a small grid ===")
for N in [3, 4, 5]:
    for c in [20, 50, 200]:
        t0 = time.time()
        result = C(ZETA, N, c)
        dt = time.time() - t0
        print(f"N={N:2d} c={c:4d} -> {result['verdict']:12s} "
              f"min_pivot={result['min_pivot']:>14s}  ({dt:.1f}s)  [{result['reason']}]")

print("\n=== sample matrix + pivots, N=4, c=50 (zeta) ===")
Q, a_list = galerkin_matrix(ZETA, 4, 50)
for i in range(4):
    print("  ".join(mp.nstr(Q[i, j], 6) for j in range(4)))
pivots, _growth = ldlt(Q)
print("pivots:", [mp.nstr(p, 6) for p in pivots])

print("\n=== C(L(s,chi_3), N, c) for comparison ===")
for N in [3, 4]:
    for c in [20, 50]:
        result = C(CHI3, N, c)
        print(f"N={N:2d} c={c:4d} -> {result['verdict']:12s} min_pivot={result['min_pivot']}")
