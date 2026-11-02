import numpy as np
from scipy.integrate import solve_ivp

"""
Lossless coupled-mode models; z in micrometers, beta/kappa in 1/micrometers.
Assuming that Kab = Kba .
"""

def delta(beta_a, beta_b, Kaa =0, Kbb=0):
    """
    Half of the effective phase mismatch.

    beta_a : Propagation constant of mode a
    beta_b : Propagation constant of mode b
    Kaa : Self-coupling coefficient of mode a (default: 0)
    Kbb : Self-coupling coefficient of mode b (default: 0)
    """
    return 0.5 * (beta_a - beta_b + Kaa - Kbb)


def hamiltonian(delta, kappa):
    """
    Effective coupled-mode Hamiltonian.

    delta : Half of the effective phase mismatch
    kappa : Coupling coefficient
    """
    return np.array([[delta, kappa], [np.conj(kappa), -delta]], dtype=complex)

def propagate(z, delta=0., kappa=0.1, initial=(1., 0.), rtol=1e-10, atol=1e-12):
    """
    Propagate the coupled-mode equations.

    z : Array of positions
    delta : Half of the effective phase mismatch
    kappa : Coupling coefficient
    initial : Initial amplitudes. default: (1, 0)
    rtol : Relative tolerance
    atol : Absolute tolerance
    """
    z = np.asarray(z, dtype=float)
    H = hamiltonian(delta, kappa)
    initial = np.asarray(initial, dtype=complex)
    if not np.isclose(np.vdot(initial, initial).real, 1):
        raise ValueError('Initial amplitudes must have unit total power')
    sol = solve_ivp(lambda x, a: -1j * H @ a, (z[0], z[-1]), initial, t_eval=z, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y

def analytic_powers(z, kappa, delta=0.):
    """
    Analytic solution for the powers in the coupled-mode equations.
    For Lossless Coupled Mode Theory.

    z : Array of positions
    kappa : Coupling coefficient
    delta : Half of the effective phase mismatch
    """
    z = np.asarray(z)
    omega = np.sqrt(kappa**2 + delta**2)
    p2 = np.zeros_like(z, dtype=float) if omega == 0 else (kappa / omega)**2 * np.sin(omega*z)**2
    return 1-p2, p2

def coupling_length(kappa):
    return np.pi/(2*abs(kappa))



"""Geometry-dependent coupled-mode extension (separate from original coupled_mode.py).

Based on isolated symmetric 1D TE0 slab modes, weak-coupling scalar CMT.
Lengths in um; beta, kappa, delta in um^-1.
No user files are modified or overwritten.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import trapezoid

# Example material parameters, NOT calibrated silicon strip waveguides.
lambda0 = 1.55
n_core = 1.50
n_clad = 1.45
k0 = 2*np.pi/lambda0

def effective_delta(beta_a, beta_b, Kaa=0.0, Kbb=0.0):
    """Identical to the user's delta definition, with a distinct name to avoid clashes."""
    return 0.5 * (beta_a-beta_b+Kaa-Kbb)

def powers_from_coefficients(z, kappa, delta_value=0.0):
    """Lossless analytical powers for initial amplitudes (1,0)."""
    omega=np.hypot(np.abs(kappa), delta_value)
    z=np.asarray(z, dtype=float)
    pb=np.zeros_like(z) if omega == 0 else (np.abs(kappa)/omega)**2*np.sin(omega*z)**2
    return 1-pb,pb

def te0_slab_mode(width, x):
    """Normalized fundamental even TE slab mode; x centered on a guide [um]."""
    if width <= 0:
        raise ValueError('Width must be positive')
    V2 = (k0**2) * (n_core**2 - n_clad**2)
    # u=h*w/2, v=q*w/2, u*tan(u)=v, u^2+v^2=V^2.
    V = 0.5 * width * np.sqrt(V2)
    f = lambda u: u*np.tan(u) - np.sqrt(max(V*V-u*u, 0.0))
    u = brentq(f, 1e-12, min(V, np.pi/2-1e-9))
    h = 2*u/width
    q = np.sqrt(max(V2-h*h, 0.0))
    beta = np.sqrt((k0*n_clad)**2+q*q)
    r = np.abs(x)
    e = np.where(r <= width/2, np.cos(h*r), np.cos(h*width/2)*np.exp(-q*(r-width/2)))
    return beta, q, e

def slab_cmt_geometry(width_a, width_b, gap, dx=0.015):
    """Approximate CMT coefficients using isolated TE0 slab modes.

    Return beta_a,b, self-shifts Kaa,Kbb, symmetric effective kappa, delta.
    Modal overlap of the perturbation in the *other* core; weak coupling only.
    """
    if gap < 0: raise ValueError('Gap must be nonnegative')
    center_sep = gap + (width_a+width_b)/2
    xa, xb = -center_sep/2, center_sep/2
    xspan = max(18.0, center_sep/2+12)
    x = np.arange(-xspan, xspan+dx, dx)
    beta_a, q_a, ea = te0_slab_mode(width_a, x-xa)
    beta_b, q_b, eb = te0_slab_mode(width_b, x-xb)
    ea /= np.sqrt(trapezoid(ea**2, x))
    eb /= np.sqrt(trapezoid(eb**2, x))
    core_a = np.abs(x-xa) <= width_a/2
    core_b = np.abs(x-xb) <= width_b/2
    contrast = n_core**2-n_clad**2
    # Scalar CMT first-order self-shifts; the guides' index perturbations are disjoint.
    Kaa = (k0*k0*contrast/(2*beta_a))*trapezoid(ea**2*core_b, x)
    Kbb = (k0*k0*contrast/(2*beta_b))*trapezoid(eb**2*core_a, x)
    Kab = (k0*k0*contrast/(2*beta_a))*trapezoid(ea*eb*core_b, x)
    Kba = (k0*k0*contrast/(2*beta_b))*trapezoid(ea*eb*core_a, x)
    # Identical guides have equal self-coupling by mirror symmetry; remove grid quadrature noise.
    if np.isclose(width_a, width_b, rtol=0, atol=1e-12):
        Kaa = Kbb = 0.5*(Kaa+Kbb)
    # Hermitian effective coupling after amplitude re-normalization in this approximation.
    kappa = np.sqrt(max(Kab*Kba, 0))
    return dict(beta_a=beta_a,beta_b=beta_b,q_a=q_a,q_b=q_b,
                Kaa=Kaa,Kbb=Kbb,kappa=kappa,
                delta=effective_delta(beta_a,beta_b,Kaa,Kbb),d=center_sep)



def run_geometry_sweeps():
    """Generate width sweep, gap sweep, and selected propagation trajectories."""
    width_b = 1.5   # um, fixed
    fixed_gap = 0.40  # um, fixed
    width_a_values = np.linspace(0.8, 2.2, 45)
    L = 150.0  # fixed coupler length [um]

    sweep_w = [slab_cmt_geometry(w, width_b, fixed_gap) for w in width_a_values]
    k_w = np.array([s['kappa'] for s in sweep_w])
    d_w = np.array([s['delta'] for s in sweep_w])
    beta_w = np.array([s['beta_a'] for s in sweep_w])
    max_w = k_w**2/(k_w**2+d_w**2)
    P_out_w = [powers_from_coefficients(L, k, d)[1] for k,d in zip(k_w,d_w)]

    fig, ax = plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(width_a_values,k_w)
    ax[0,0].set(ylabel=r'$\kappa$ [$\mu m^{-1}$]',title='Coupling vs width of guide a')
    ax[0,1].plot(width_a_values,d_w)
    ax[0,1].axhline(0,color='gray',ls=':')
    ax[0,1].set(ylabel=r'$\delta$ [$\mu m^{-1}$]',title='Effective phase mismatch')
    ax[1,0].plot(width_a_values,beta_w,label=r'$\beta_a$')
    ax[1,0].axhline(sweep_w[0]['beta_b'],ls='--',label=r'$\beta_b$')
    ax[1,0].set(ylabel=r'$\beta$ [$\mu m^{-1}$]',title='Isolated propagation constants')
    ax[1,0].legend()
    ax[1,1].plot(width_a_values,max_w,label=r'$P_{b,max}$')
    ax[1,1].plot(width_a_values,P_out_w,label=rf'$P_b(L={L:g}\ \mu m)$')
    ax[1,1].set(ylabel='Normalized power',ylim=(-0.03,1.03),title='Transfer efficiency')
    ax[1,1].legend()
    for a in ax.flat:
        a.set_xlabel(r'Width $w_a$ [$\mu m$]')
        a.grid(alpha=.35)
    plt.tight_layout()
    plt.show()

    width_a = width_b = 1.5
    edge_gaps = np.linspace(0.15, 2.0, 45)
    sweep_g = [slab_cmt_geometry(width_a,width_b,g) for g in edge_gaps]
    k_g = np.array([s['kappa'] for s in sweep_g])
    d_g = np.array([s['delta'] for s in sweep_g])
    Lc = np.pi/(2*k_g)
    Pout_g = [powers_from_coefficients(L,k,d)[1] for k,d in zip(k_g,d_g)]

    fig, ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].semilogy(edge_gaps,k_g)
    ax[0].set(ylabel=r'$\kappa$ [$\mu m^{-1}$]',title='Coupling vs gap')
    ax[1].plot(edge_gaps,Lc)
    ax[1].set(ylabel=r'$L_c$ [$\mu m$]',title='First complete-transfer length')
    ax[2].plot(edge_gaps,Pout_g)
    ax[2].set(ylabel=rf'$P_b(L={L:g}\ \mu m)$',ylim=(-0.03,1.03),title='Power at fixed length')
    for a in ax:
        a.set_xlabel(r'Edge gap $g$ [$\mu m$]')
        a.grid(alpha=.35)
    plt.tight_layout()
    plt.show()
    print('Maximum |delta| for identical guides:',np.max(np.abs(d_g)))

    z = np.linspace(0,L,1200)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for gap in (0.2,0.5,1.0):
        pars=slab_cmt_geometry(1.5,1.5,gap)
        pa,pb=powers_from_coefficients(z,pars['kappa'],pars['delta'])
        ax.plot(z,pb,label=rf'$g={gap:.1f}\ \mu m$, $\kappa={pars["kappa"]:.4f}$')
        assert np.allclose(pa+pb,1)
    ax.set(xlabel=r'$z$ [$\mu m$]',ylabel=r'$P_b(z)$',title='Gap changes the beating period',ylim=(-0.03,1.03))
    ax.legend()
    ax.grid(alpha=.35)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample=slab_cmt_geometry(1.5,1.5,0.4)
    print("Check: identical guides yield delta =",sample["delta"])
    assert abs(sample["delta"]) < 1e-10
    run_geometry_sweeps()
