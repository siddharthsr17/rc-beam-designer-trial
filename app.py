import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import brentq

st.set_page_config(page_title="RC Section Design", layout="wide")

# --- 1. CORE CALCULATION ENGINE ---
def analyze_section(b, h, c_nom, bar_dia_tens, num_bars_tens, bar_dia_comp, 
                    num_bars_comp, link_dia, link_spacing, num_legs, 
                    fck, fyk, Es, M_ed, V_ed, M_char, M_freq, M_qp, w_max_req):
    
    # Material Constants & Safety Factors
    gamma_c, gamma_s = 1.5, 1.15
    fcd = 0.85 * fck / gamma_c
    fyd = fyk / gamma_s
    fctm = 0.30 * (fck ** (2/3)) if fck <= 50 else 2.12 * math.log(1 + (fck / 10.0))
    Ecm = 22000.0 * ((fck + 8) / 10)**0.3
    
    # Effective Depths & Areas
    As_tens = num_bars_tens * (math.pi * bar_dia_tens**2 / 4.0)
    As_comp = num_bars_comp * (math.pi * bar_dia_comp**2 / 4.0)
    d_tens = h - c_nom - link_dia - (bar_dia_tens / 2.0)
    d_comp = c_nom + link_dia + (bar_dia_comp / 2.0)
    d = d_tens
    
    # --- ULS BENDING ---
    lambda_pb, eta_pb = 0.8, 1.0
    
    def uls_force_equilibrium(x):
        Fc = eta_pb * fcd * b * (lambda_pb * x)
        eps_s = 0.0035 * (d - x) / x
        Fs_tens = As_tens * min(eps_s * Es, fyd)
        
        if As_comp > 0:
            eps_sc = 0.0035 * (x - d_comp) / x
            Fs_comp = As_comp * max(min(eps_sc * Es, fyd), -fyd)
        else:
            Fs_comp = 0.0
        return Fc + Fs_comp - Fs_tens

    try:
        x_uls = brentq(uls_force_equilibrium, 1.0, d)
    except ValueError:
        x_uls = d # Failsafe if over-reinforced

    Fc_uls = eta_pb * fcd * b * (lambda_pb * x_uls)
    eps_sc_uls = 0.0035 * (x_uls - d_comp) / x_uls if As_comp > 0 else 0
    fs_comp_uls = max(min(eps_sc_uls * Es, fyd), -fyd) if As_comp > 0 else 0
    
    M_rd = (Fc_uls * (d - 0.5 * lambda_pb * x_uls) + As_comp * fs_comp_uls * (d - d_comp)) / 1e6
    uls_bending_pass = M_rd >= M_ed
    
    # --- ULS SHEAR ---
    k_shear = min(1.0 + math.sqrt(200.0 / d), 2.0)
    rho_l = min(As_tens / (b * d), 0.02)
    CRd_c = 0.18 / gamma_c
    v_min = 0.035 * (k_shear**1.5) * (fck**0.5)
    VRd_c = max((CRd_c * k_shear * ((100 * rho_l * fck)**(1/3))) * b * d, v_min * b * d) / 1e3
    
    cot_theta, z = 2.5, 0.9 * d
    Asw = num_legs * (math.pi * link_dia**2 / 4.0)
    VRd_s = (Asw / link_spacing) * z * fyd * cot_theta / 1e3
    nu1 = 0.6 * (1.0 - fck / 250.0)
    VRd_max = (1.0 * b * z * nu1 * fcd / (cot_theta + 1.0/cot_theta)) / 1e3
    
    VRd = min(VRd_s, VRd_max)
    uls_shear_pass = VRd >= V_ed

    # --- SLS STRESS & CRACKING ---
    phi_eff = 2.0
    alpha_e = Es / (Ecm / (1.0 + phi_eff))
    
    A_a = 0.5 * b
    B_a = alpha_e * (As_tens + As_comp)
    C_a = -alpha_e * (As_tens * d + As_comp * d_comp)
    x_sls = (-B_a + math.sqrt(B_a**2 - 4 * A_a * C_a)) / (2 * A_a)
    
    I_cr = (b * (x_sls**3) / 3.0) + alpha_e * As_tens * ((d - x_sls)**2) + alpha_e * As_comp * ((x_sls - d_comp)**2)
    
    sigma_c_char = (M_char * 1e6 / I_cr) * x_sls
    sigma_s_char = alpha_e * (M_char * 1e6 / I_cr) * (d - x_sls)
    sigma_c_qp = (M_qp * 1e6 / I_cr) * x_sls
    sigma_s_qp = alpha_e * (M_qp * 1e6 / I_cr) * (d - x_sls)
    
    lim_c_char, lim_s_char, lim_c_qp = 0.60 * fck, 0.80 * fyk, 0.45 * fck
    
    h_c_eff = min(2.5 * (h - d), (h - x_sls) / 3.0, h / 2.0)
    rho_p_eff = As_tens / (b * h_c_eff)
    S_rmax = 3.4 * c_nom + (0.8 * 0.5 * 0.425 * bar_dia_tens) / rho_p_eff
    eps_diff = max((sigma_s_qp - 0.4 * (fctm / rho_p_eff) * (1.0 + alpha_e * rho_p_eff)) / Es, 0.6 * (sigma_s_qp / Es))
    w_k = S_rmax * eps_diff
    
    return locals()

# --- 2. STREAMLIT UI ---
st.title("Eurocode 2: RC Section Design")

# SIDEBAR INPUTS
with st.sidebar:
    st.header("Section Geometry (mm)")
    b = st.number_input("Width (b)", 200, 1000, 300, 50)
    h = st.number_input("Depth (h)", 200, 2000, 600, 50)
    c_nom = st.number_input("Nominal Cover (c_nom)", 20, 100, 40, 5)
    
    st.header("Reinforcement")
    st.subheader("Tension (Bottom)")
    num_bars_tens = st.number_input("Count (Tens)", 2, 10, 4)
    bar_dia_tens = st.selectbox("Dia (Tens)", [12.0, 16.0, 20.0, 25.0, 32.0], index=3)
    
    st.subheader("Compression (Top)")
    num_bars_comp = st.number_input("Count (Comp)", 0, 10, 2)
    bar_dia_comp = st.selectbox("Dia (Comp)", [12.0, 16.0, 20.0, 25.0, 32.0], index=1)
    
    st.subheader("Shear Links")
    link_dia = st.selectbox("Link Dia", [8.0, 10.0, 12.0, 16.0], index=1)
    link_spacing = st.number_input("Spacing", 50, 400, 150, 25)
    num_legs = st.number_input("Legs", 2, 6, 2, 2)
    
    st.header("Material & Loads")
    fck = st.number_input("Concrete fck (MPa)", 20.0, 60.0, 30.0, 5.0)
    M_ed = st.number_input("M_ed (ULS Bending, kNm)", 0.0, 2000.0, 350.0)
    V_ed = st.number_input("V_ed (ULS Shear, kN)", 0.0, 1000.0, 180.0)
    M_char = st.number_input("M_char (SLS, kNm)", 0.0, 2000.0, 220.0)
    M_qp = st.number_input("M_qp (SLS, kNm)", 0.0, 2000.0, 140.0)

# RUN CALCULATION
res = analyze_section(b, h, c_nom, bar_dia_tens, num_bars_tens, bar_dia_comp, num_bars_comp, 
                      link_dia, link_spacing, num_legs, fck, 500.0, 200000.0, M_ed, V_ed, 
                      M_char, M_char*0.8, M_qp, 0.3)

# MAIN DASHBOARD PANELS
col1, col2, col3, col4 = st.columns(4)
col1.metric("ULS Bending", f"{res['M_rd']:.1f} kNm", f"Req: {M_ed}", delta_color="normal" if res['uls_bending_pass'] else "inverse")
col2.metric("ULS Shear", f"{res['VRd']:.1f} kN", f"Req: {V_ed}", delta_color="normal" if res['uls_shear_pass'] else "inverse")
col3.metric("Max Crack Width", f"{res['w_k']:.3f} mm", "Req: 0.30 mm", delta_color="normal" if res['w_k']<=0.3 else "inverse")
col4.metric("Concrete Stress (Char)", f"{res['sigma_c_char']:.1f} MPa", f"Lim: {res['lim_c_char']:.1f}", delta_color="normal" if res['sigma_c_char']<=res['lim_c_char'] else "inverse")

st.markdown("---")

col_draw, col_report = st.columns([1, 1.5])

with col_draw:
    st.subheader("Cross Section Visual")
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(-50, b + 50)
    ax.set_ylim(-50, h + 50)
    
    # Concrete boundary
    ax.add_patch(patches.Rectangle((0, 0), b, h, fill=True, color='#E0E0E0', ec='black', lw=2))
    
    # Shear Link
    link_w = b - 2*c_nom
    link_h = h - 2*c_nom
    ax.add_patch(patches.Rectangle((c_nom, c_nom), link_w, link_h, fill=False, ec='red', lw=2, linestyle='--'))
    
    # Tension Bars (Bottom)
    if num_bars_tens > 0:
        spacing_t = (link_w - bar_dia_tens) / (num_bars_tens - 1) if num_bars_tens > 1 else 0
        start_x_t = c_nom + link_dia + bar_dia_tens/2
        y_t = c_nom + link_dia + bar_dia_tens/2
        for i in range(int(num_bars_tens)):
            ax.add_patch(patches.Circle((start_x_t + i*spacing_t, y_t), bar_dia_tens/2, color='black'))
            
    # Compression Bars (Top)
    if num_bars_comp > 0:
        spacing_c = (link_w - bar_dia_comp) / (num_bars_comp - 1) if num_bars_comp > 1 else 0
        start_x_c = c_nom + link_dia + bar_dia_comp/2
        y_c = h - (c_nom + link_dia + bar_dia_comp/2)
        for i in range(int(num_bars_comp)):
            ax.add_patch(patches.Circle((start_x_c + i*spacing_c, y_c), bar_dia_comp/2, color='black'))
            
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)

with col_report:
    st.subheader("Detailed Engineering Report")
    st.markdown(f"""
    **Section Properties:**
    * Effective Depth ($d$): **{res['d']:.1f} mm**
    * Tensile Rebar ($A_s$): **{res['As_tens']:.0f} mm²**
    * Compressive Rebar ($A_{{s,2}}$): **{res['As_comp']:.0f} mm²**
    
    **Ultimate Limit State (ULS):**
    * NA Depth at ULS ($x$): **{res['x_uls']:.1f} mm**
    * Bending Capacity ($M_{{Rd}}$): **{res['M_rd']:.1f} kNm**
    * Concrete Shear Capacity ($V_{{Rd,c}}$): **{res['VRd_c']:.1f} kN**
    * Shear Link Capacity ($V_{{Rd,s}}$): **{res['VRd_s']:.1f} kN** (Critical: **{res['VRd']:.1f} kN**)
    
    **Serviceability Limit State (SLS):**
    * NA Depth at SLS ($x$): **{res['x_sls']:.1f} mm**
    * Steel Tensile Stress ($Characteristic$): **{res['sigma_s_char']:.1f} MPa** *(Limit: {res['lim_s_char']:.1f} MPa)*
    * Concrete Compressive Stress ($QP$): **{res['sigma_c_qp']:.1f} MPa** *(Limit: {res['lim_c_qp']:.1f} MPa)*
    
    **Cracking (Quasi-Permanent):**
    * Effective Embedment Depth ($h_{{c,eff}}$): **{res['h_c_eff']:.1f} mm**
    * Max Crack Spacing ($S_{{r,max}}$): **{res['S_rmax']:.1f} mm**
    * Calculated Crack Width ($w_k$): **{res['w_k']:.3f} mm**
    """)
