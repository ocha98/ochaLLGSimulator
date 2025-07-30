import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from math import sin, cos, radians

# physical constants
# reference: https://physics.nist.gov/cgi-bin/cuu/Value?gammae (2025/07/17)
GAMMA_E =  1.760_859_627_84e11 # gyromagnetic ratio of electron [s^-1 T^-1]
# # reference: https://physics.nist.gov/cgi-bin/cuu/Value?mu0 (2025/07/17)
MU_0 = 1.256_637_016_27e-6 # vacuum magnetic permeability [N/A^2]
# reference: https://physics.nist.gov/cgi-bin/cuu/Value?hbar (2025/07/18)
H_BAR = 1.054_571_817e-34 # reduced Planck constant [Js]
# reference: https://physics.nist.gov/cgi-bin/cuu/Value?e (2025/07/18)
E = 1.602_176_634e-19 # elementary charge [C]

# SI constants
NANO = 1e-9

# h_eff [T]
def precession_torque(m, h_eff):
    return -GAMMA_E * np.cross(m, h_eff)

# h_eff [T]
def damping_torque(m, h_eff, alpha):
    return -alpha*GAMMA_E * np.cross(m, np.cross(m, h_eff))

# ms [A/m] tf[m] jc [A/m^2]
def sot(m, theta_sh, ms, tf, jc, alpha, sigma):
    coeff = GAMMA_E*H_BAR*theta_sh / (2*E*ms*tf) * jc
    return coeff * (np.cross(m, np.cross(sigma, m)) - alpha*np.cross(sigma, m))

# ku[J/m^3], ms[A/m]
def anisotropy_field(m, ku, ms, u_axis):
    return 2 * ku / ms * np.dot(m, u_axis) * u_axis

# hext: 外部磁場 [T]
def llg(t, m, hext, alpha, ku, ms, u_axis, theta_sh, tf, jc, sigma):
    jc = jc(t)
    total_heff = hext + anisotropy_field(m, ku, ms, u_axis)

    dm_dt = precession_torque(m, total_heff) + damping_torque(m, total_heff, alpha) + sot(m, theta_sh, ms, tf, jc, alpha, sigma)

    return 1 / (1 + alpha*alpha) * dm_dt

def simulator(run_time, dt = None) -> pd.DataFrame:
    # params
    hext = np.array([0, 0, 0]) # T
    alpha = 0.05
    ku = 1e6 # [J/m^3]
    ms = 1e6 # [A/m]
    u_axis = np.array([0, 0, 1])
    theta_sh = 0.07
    tf = 1e-9
    jc = lambda t: 1e12 # [A/m^2] (constant current density)
 
    sigma = np.array([0, 0, -1])

    theta = radians(1)
    phi = radians(0)
    m_init = np.array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]) # 初期磁化

    t_list = None
    if dt is not None:
        t_list = np.arange(0, run_time, run_time/dt)

    result = solve_ivp(llg, (0, run_time), m_init, t_eval = t_list, args = (hext, alpha, ku, ms, u_axis, theta_sh, tf, jc, sigma)) # 数値計算実行！

    # == 計算結果をプロット == 
    t_list = result.t
    mx = result.y[0]
    my = result.y[1]
    mz = result.y[2]

    df = pd.DataFrame()
    df['t'] = t_list
    df['mx'] = mx
    df['my'] = my
    df['mz'] = mz
    df['jc'] = list(map(jc, t_list))

    return df

def plot_result(df: pd.DataFrame):
    t_list = df['t'].values
    mx = df['mx'].values
    my = df['my'].values
    mz = df['mz'].values
    jc = df['jc'].values
    
    fig = plt.figure(layout = 'constrained')
    fig.suptitle('Simulation Result', fontsize = 16)
    # 3D軌道をプロット
    ax1 = fig.add_subplot(122, projection = '3d')
    
    # xyzの単位ベクトルを描画
    ax1.quiver(0, 0, 0, 1, 0, 0, color = 'k', linestyle = '--', linewidth = 0.5) # x arrow
    ax1.quiver(0, 0, 0, 0, 1, 0, color = 'k', linestyle = '--', linewidth = 0.5) # y arrow
    ax1.quiver(0, 0, 0, 0, 0, 1, color = 'k', linestyle = '--', linewidth = 0.5) # z arrow
    ax1.text(1.2, 0, 0, 'x', ha = 'left')
    ax1.text(0, 1.5, 0, 'y', ha = 'left')
    ax1.text(0, 0, 1.5, 'z', ha = 'left')

    # 軌道をプロット
    ax1.plot(mx, my, mz, marker = '')
    
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    # 枠線などを消す
    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis.line.set_color('none')
    ax1.yaxis.line.set_color('none')
    ax1.zaxis.line.set_color('none')
    ax1.xaxis.pane.set_edgecolor('none')
    ax1.yaxis.pane.set_edgecolor('none')
    ax1.zaxis.pane.set_edgecolor('none')

    # 補助線を描画
    for theta in np.linspace(0, np.pi, 5):
        r = np.sin(theta)
        lin = np.linspace(0, 2*np.pi, 100)
        ax1.plot(r*np.cos(lin), r*np.sin(lin), np.cos(theta), color = 'gray', marker = '', linewidth = 0.5)
    for theta in np.linspace(0, np.pi, 5):
        lin = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta)*np.sin(lin), np.sin(theta)*np.sin(lin), np.cos(lin), color = 'gray', marker = '', linewidth = 0.5)

    ax1.text(mx[0], my[0], mz[0], 'Initial', ha = 'right', va = 'bottom')
    ax1.plot(mx[0], my[0], mz[0], marker = 'o', color = 'red')
    ax1.text(mx[len(mx)-1], my[len(my)-1], mz[len(mz)-1], 'Final', ha = 'left', va = 'bottom')
    ax1.plot(mx[len(mx)-1], my[len(my)-1], mz[len(mz)-1], marker = 'o', color = 'red')

    # mx, my, mzをプロット
    ax = fig.add_subplot(421)
    ax.plot(t_list/NANO, jc)
    ax.set_ylabel('jc (A/m^2)')
    ax.minorticks_on()

    ax2 = fig.add_subplot(423)
    ax2.plot(t_list/NANO, mx)
    ax2.set_ylabel('mx')
    ax2.set_ylim(-1.1, 1.1)
    ax2.minorticks_on()

    ax3 = fig.add_subplot(425)
    ax3.plot(t_list/NANO, my)
    ax3.set_ylabel('my')
    ax3.set_ylim(-1.1, 1.1)
    ax3.minorticks_on()

    ax4 = fig.add_subplot(427)
    ax4.plot(t_list/NANO, mz)
    ax4.set_ylabel('mz')
    ax4.set_xlabel('t (ns)')
    ax4.set_ylim(-1.1, 1.1)
    ax4.minorticks_on()

    fig.savefig('simulation_result.jpg', dpi = 200)
    plt.close(fig)

def main():
    run_time = 1.5e-9  # 1.5 ns
    dt = 1000  # 時間刻み数

    df = simulator(run_time, dt)

    plot_result(df)

if __name__ == '__main__':
    main()
