import mpmath as mp
from sympy import primerange

mp.mp.dps = 40

# First 10 nontrivial zeta zero imaginary parts (standard tabulated values)
GAMMA = [mp.mpf(s) for s in [
"14.134725141734693790",
"21.022039638771554993",
"25.010857580145688763",
"30.424876125859513210",
"32.935061587739189691",
"37.586178158825671257",
"40.918719012147495187",
"43.327073280914999519",
"48.005150881167159727",
"49.773832477672302181",
]]

def h(r, a):
    return mp.e**(-a*r*r)

def g(u, a):
    return 1/(2*mp.sqrt(mp.pi*a)) * mp.e**(-u*u/(4*a))

def lhs(a, zeros=GAMMA):
    return 2*sum(h(gm, a) for gm in zeros)

def rhs(a, prime_limit=20000):
    pole = 2*mp.e**(a/4)
    def integrand(r):
        return h(r, a) * mp.re(mp.digamma(mp.mpf('0.25') + 1j*r/2))
    R = mp.sqrt(80/a)  # generous cutoff where h(R) is astronomically small
    arch = (1/(2*mp.pi)) * mp.quad(integrand, [-R, 0, R])
    arch -= g(0, a) * mp.log(mp.pi)

    prime_sum = mp.mpf(0)
    for p in primerange(2, prime_limit):
        logp = mp.log(p)
        k = 1
        while True:
            val = logp * mp.mpf(p)**(-mp.mpf(k)/2) * g(k*logp, a)
            if val < mp.mpf('1e-35') and k > 1:
                break
            prime_sum += val
            k += 1
            if k > 60:
                break

    return pole + arch - 2*prime_sum

print(f"{'a':>8} {'LHS (10 known zeros)':>28} {'RHS (arch+pole-primes)':>28} {'rel.diff':>12}")
for a in ["0.02", "0.01", "0.005"]:
    a_mp = mp.mpf(a)
    L = lhs(a_mp)
    R = rhs(a_mp)
    reldiff = abs(L-R)/abs(R)
    print(f"{a:>8} {mp.nstr(L,15):>28} {mp.nstr(R,15):>28} {mp.nstr(reldiff,4):>12}")
