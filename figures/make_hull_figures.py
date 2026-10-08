"""Wigley hull 3D figures: geometry render, panel mesh, Cp from a low-order
Rankine source panel method (double-body potential flow), and flow streamlines.
Honest numerical illustrations -- not commercial CFD (RANS) results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / 'figures'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 160, 'font.size': 10,
                     'axes.titlesize': 12})

# Wigley hull: L=1, B=0.1, T=0.0625
L, B, T = 1.0, 0.1, 0.0625
U = 1.0  # inflow speed in +x


def half_breadth(x, z):
    """y = f(x, z), half hull (y >= 0)."""
    return (B / 2) * (1 - (2 * x / L) ** 2) * (1 - (z / T) ** 2)


def hull_grid(nx, nz, ztop=0.0):
    """Cell-centered parametric grid on half hull (y>=0). Returns X, Y, Z (nz, nx)."""
    xs = np.linspace(-L / 2, L / 2, nx)
    zs = np.linspace(-T, ztop, nz)
    X, Z = np.meshgrid(xs, zs)
    Y = half_breadth(X, Z)
    return X, Y, Z


def panels_from_grid(X, Y, Z):
    """Quadrilateral panels from node grid -> centroids, normals, areas."""
    P = np.stack([X, Y, Z], axis=-1)  # (nz, nx, 3)
    p00 = P[:-1, :-1]; p10 = P[:-1, 1:]; p01 = P[1:, :-1]; p11 = P[1:, 1:]
    C = (p00 + p10 + p01 + p11) / 4
    d1 = p11 - p00; d2 = p01 - p10
    N = np.cross(d1, d2)
    A = np.linalg.norm(N, axis=-1) / 2
    n = N / np.linalg.norm(N, axis=-1, keepdims=True)
    # orient outward (+y side, and +x at bow): flip if pointing inward
    out = np.stack([np.zeros_like(n[..., 0]), np.ones_like(n[..., 0]),
                    np.zeros_like(n[..., 0])], axis=-1)
    flip = (np.sum(n * out, axis=-1) < 0)
    n[flip] *= -1
    return C.reshape(-1, 3), n.reshape(-1, 3), A.reshape(-1)


def solve_sources(C, N, A):
    """Low-order constant-source panel method, double-body (z-mirror) + y-symmetry.
    Returns source strengths sigma (M,)."""
    M = len(C)
    Cz, Nz = C * [1, 1, -1], N * [1, 1, -1]
    Cy, Ny = C * [1, -1, 1], N * [1, -1, 1]
    sets = [(C, N), (Cz, Nz), (Cy, Ny), (Cz * [1, -1, 1], Nz * [1, -1, 1])]
    # influence matrix: normal velocity at i from unit-strength panel j
    Amat = np.zeros((M, M))
    for Cj, Nj in sets:
        D = C[:, None, :] - Cj[None, :, :]          # (M, M, 3)
        R = np.linalg.norm(D, axis=-1)              # (M, M)
        np.fill_diagonal(R, np.inf)                 # self handled separately
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.sum(D * N[:, None, :], axis=-1) / (4 * np.pi * R ** 3)  # (M,M)
        K[~np.isfinite(K)] = 0.0
        Amat += K * A[None, :]
    np.fill_diagonal(Amat, Amat.diagonal() + 0.5)   # self-induced normal velocity
    b = -U * N[:, 0]
    return np.linalg.solve(Amat, b)


def velocity_field(P, C, N, A, sigma, chunk=600):
    """Velocity at field points P (K,3) from uniform flow + sources (+images)."""
    sets_C = [C, C * [1, 1, -1], C * [1, -1, 1], C * [1, -1, -1]]
    V = np.zeros_like(P)
    V[:, 0] = U
    w = sigma * A / (4 * np.pi)
    for Cj in sets_C:
        for s in range(0, len(P), chunk):
            D = P[s:s + chunk, None, :] - Cj[None, :, :]
            R = np.linalg.norm(D, axis=-1, keepdims=True)
            R = np.maximum(R, 1e-9)
            V[s:s + chunk] += np.sum(w[None, :, None] * D / R ** 3, axis=1)
    return V


def surface_velocity(C, N, A, sigma):
    """Tangential surface velocity at centroids (self tangential = 0)."""
    sets_C = [C, C * [1, 1, -1], C * [1, -1, 1], C * [1, -1, -1]]
    Vt = np.zeros_like(C)
    Vt[:, 0] = U
    w = sigma * A / (4 * np.pi)
    M = len(C)
    for Cj in sets_C:
        D = C[:, None, :] - Cj[None, :, :]
        R = np.linalg.norm(D, axis=-1, keepdims=True)
        R = np.maximum(R, 1e-9)
        Vij = np.sum(w[None, :, None] * D / R ** 3, axis=1)
        Vij[np.arange(M), :] = 0.0  # drop self (point approx invalid); tangential self = 0
        Vt += Vij
    # remove normal component -> tangential
    Vn = np.sum(Vt * N, axis=1, keepdims=True)
    return Vt - Vn * N


def fig_geometry():
    X, Y, Z = hull_grid(81, 41)
    ls = LightSource(azdeg=315, altdeg=45)
    fig = plt.figure(figsize=(11, 4.6))
    for k, (elev, azim, title) in enumerate([(18, -62, 'Bow-quarter view'), (8, -90, 'Profile view')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        # starboard + mirrored port side
        for sgn, cmap in [(1, 'Blues'), (-1, 'Blues')]:
            rgb = ls.shade(np.ones_like(X), cmap=plt.get_cmap(cmap), vert_exag=0.1,
                           blend_mode='soft')
            ax.plot_surface(X, sgn * Y, Z, rstride=2, cstride=2, facecolors=rgb,
                            linewidth=0, antialiased=True, shade=False)
        # waterplane
        xw = np.linspace(-L / 2, L / 2, 60); yw = np.linspace(-B / 2 * 1.6, B / 2 * 1.6, 30)
        Xw, Yw = np.meshgrid(xw, yw)
        ax.plot_surface(Xw, Yw, np.zeros_like(Xw), color='#9fc8e8', alpha=0.35,
                        linewidth=0, antialiased=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((2.2, 1, 0.5))
        ax.set_axis_off()
    fig.suptitle('Wigley hull geometry (L/B = 10, B/T = 1.6) — parametric CAD-style render')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_hull_geometry.png', bbox_inches='tight')
    plt.close(fig); print('wrote fig3')


def fig_mesh():
    X, Y, Z = hull_grid(49, 25)
    fig = plt.figure(figsize=(11, 4.6))
    for k, (elev, azim, title, xs) in enumerate([
            (18, -62, 'Surface panel mesh (half hull shown)', (1,)),
            (12, -35, 'Bow region close-up', (1,))]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        if k == 0:
            ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='#1f6f9f', lw=0.4)
            ax.plot_wireframe(X, -Y, Z, rstride=2, cstride=2, color='#1f6f9f', lw=0.25, alpha=0.5)
        else:
            m = X > 0.15
            ax.plot_wireframe(np.where(m, X, np.nan), np.where(m, Y, np.nan),
                              np.where(m, Z, np.nan), color='#1f6f9f', lw=0.7)
            ax.set_xlim(0.15, 0.5); ax.set_ylim(0, 0.06); ax.set_zlim(-0.0625, 0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title); ax.set_axis_off()
    fig.suptitle('Computational mesh on Wigley hull — CFD meshing style')
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_hull_mesh.png', bbox_inches='tight')
    plt.close(fig); print('wrote fig4')


def fig_cp(sigma_data=None):
    # cosine-spaced in x to resolve the stem/stern stagnation zones
    nx, nz = 61, 31
    theta = (np.arange(nx) + 0.5) * np.pi / nx
    xs = -np.cos(theta) * L / 2
    zs = np.linspace(-T, -T / 30, nz)
    X, Z = np.meshgrid(xs, zs)
    Y = half_breadth(X, Z)
    C, N, A = panels_from_grid(X, Y, Z)
    cache = FIG / '_panel_solution.npz'
    if cache.exists():
        d = np.load(cache)
        C, N, A, sigma = d['C'], d['N'], d['A'], d['sigma']
        print('loaded cached panel solution')
    else:
        sigma = solve_sources(C, N, A)
        np.savez(cache, C=C, N=N, A=A, sigma=sigma)
    Vt = surface_velocity(C, N, A, sigma)
    Cp = 1 - np.sum(Vt ** 2, axis=1) / U ** 2
    print(f'Cp range: [{Cp.min():.4f}, {Cp.max():.4f}] (slender hull: mild variation is physical)')
    nz, nx = X.shape[0] - 1, X.shape[1] - 1
    Cpgrid = Cp.reshape(nz, nx)
    fig = plt.figure(figsize=(11, 4.8))
    vmax = float(np.ceil(Cp.max() * 250) / 250)  # tidy upper bound from data
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap('YlOrRd')
    for k, (elev, azim, title) in enumerate([(20, -62, 'Starboard'), (20, 118, 'Port (mirrored)')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        fc = cmap(norm(Cpgrid))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fc, linewidth=0,
                        antialiased=True, shade=False)
        ax.plot_surface(X, -Y, Z, rstride=2, cstride=2, facecolors=fc, linewidth=0,
                        antialiased=True, shade=False, alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title); ax.set_axis_off()
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(m, ax=fig.axes, shrink=0.7, label='Pressure coefficient Cp')
    fig.suptitle('Hull surface pressure — source-panel potential flow (double-body, slender Wigley)')
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_hull_pressure.png', bbox_inches='tight')
    plt.close(fig); print('wrote fig5')
    return C, N, A, sigma


def fig_streamlines(C, N, A, sigma):
    zplane = -T / 2
    x = np.linspace(-0.9, 1.3, 130); y = np.linspace(0.001, 0.55, 80)
    Xg, Yg = np.meshgrid(x, y)
    P = np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, zplane)], axis=1)
    V = velocity_field(P, C, N, A, sigma)
    Um = V[:, 0].reshape(Xg.shape); Vm = V[:, 1].reshape(Xg.shape)
    inside = (np.abs(Xg) < L / 2) & (Yg < half_breadth(Xg, zplane)) & (zplane < 0)
    Um = np.where(inside, np.nan, Um); Vm = np.where(inside, np.nan, Vm)
    spd = np.sqrt(Um ** 2 + Vm ** 2) / U
    lo, hi = float(np.nanmin(spd)), float(np.nanpercentile(spd, 99.5))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cf = ax.contourf(Xg, Yg, spd, levels=np.linspace(lo, hi, 25), cmap='YlGnBu', extend='max')
    ax.streamplot(x, y, Um, Vm, density=1.6, color='k', linewidth=0.6, arrowsize=0.8)
    # hull waterline section at this draft
    xh = np.linspace(-L / 2, L / 2, 200)
    ax.fill_between(xh, 0, half_breadth(xh, zplane), color='#333333', zorder=3)
    ax.set_aspect('equal'); ax.set_xlim(-0.9, 1.3); ax.set_ylim(0, 0.55)
    ax.set_xlabel('x / L'); ax.set_ylabel('y / L')
    ax.set_title('Flow around Wigley hull at z = -T/2 — streamlines & |V|/U (panel method)')
    fig.colorbar(cf, ax=ax, label='|V| / U')
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_flow_streamlines.png', bbox_inches='tight')
    plt.close(fig); print('wrote fig6')




def fig_allocation_3d():
    """3D bar chart of the core innovation: adaptive L2-budget allocation
    across hydrodynamics / propulsion modules by policy (headroom 0.90).
    Data: data/summary.csv (project's own 2048-run statistics)."""
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'summary.csv')
    d = df[df['headroom'] == 0.9].copy()
    order = ['Fixed-L1', 'Fixed-L2', 'Threshold-Projected', 'Greedy-Online',
             'Switch-Aware-Greedy', 'Discounted-LinUCB', 'Contextual-Thompson',
             'Proposed-Exact-Online']
    short = ['Fixed-L1', 'Fixed-L2', 'Thr.-Proj.', 'Greedy', 'Sw.-Aware', 'LinUCB', 'C-Thomp.', 'Prop.-Exact']
    d['policy'] = pd.Categorical(d['policy'], order)
    d = d.sort_values('policy').reset_index(drop=True)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(len(d))
    for j, (col, color, ypos, lab) in enumerate([
            ('hydro_l2_share_mean', '#2e86c1', 0, 'Hydrodynamics'),
            ('propulsion_l2_share_mean', '#e67e22', 1, 'Propulsion')]):
        ax.bar3d(xs - 0.22 + j * 0.44 - 0.0, [ypos] * len(d), np.zeros(len(d)),
                 0.4, 0.7, d[col].values, color=color, alpha=0.88, shade=True)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=8, rotation=18, ha='right')
    ax.set_yticks([0.35, 1.35]); ax.set_yticklabels(['Hydrodynamics', 'Propulsion'])
    ax.set_zlabel('High-fidelity (L2) share', fontsize=10)
    ax.set_zlim(0, 1.15)
    ax.view_init(elev=22, azim=-58)
    ax.set_title('Adaptive fidelity allocation across modules by policy (headroom 0.90)\n'
                 'Innovation: policies co-schedule the L2 budget between modules '
                 '(Prop.-Exact: 25% hydro / 72% propulsion)', fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(FIG / 'fig7_allocation_3d.png', bbox_inches='tight')
    plt.close(fig); print('wrote fig7')

if __name__ == '__main__':
    fig_geometry()
    fig_mesh()
    C, N, A, sigma = fig_cp()
    fig_streamlines(C, N, A, sigma)
    fig_allocation_3d()
    print('done')
