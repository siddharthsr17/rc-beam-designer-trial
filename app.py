import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import brentq

st.set_page_config(page_title="RC Section Design (Eurocodes)", layout="wide")

# =============================================================================
# 1. CORE CALCULATION ENGINE
# =============================================================================
def analyze_section(b, h, c_nom, bar_dia_tens, num_bars_tens, bar_dia_comp, 
                    num_bars_comp, link_dia, link_spacing, num_legs, 
                    fck, fyk, Es, M_ed, V_ed, M_char, M_freq, M_qp, w_max_req):
    
    # --- Materials & Partial Factors (Cl. 2.4.2.4 & Cl. 3.1) ---
    gamma_c, gamma_s = 1.5, 1.15
    alpha_cc = 0.85  # Irish National Annex Clause 3.1.6(1P)
    fcd = alpha_cc * fck / gamma_c
    fyd = fyk / gamma_s
    
    # Concrete Tensile Strength & Stiffness (Table 3.1)
    fctm = 0.30 * (fck ** (2/3)) if fck <= 50 else 2.12 * math.log(1 + (fck / 10.0))
    Ecm = 22000.0 * ((fck + 8) / 10)**0.3
    
    # Geometric Depths
    As_tens = num_bars_tens * (math.pi * bar_dia_tens**2 / 4.0)
    As_comp = num_bars_comp * (math.pi * bar_dia_comp**2 / 4.0)
    d = h - c_nom - link_dia - (bar_dia_tens / 2.0)
    d_comp = c_nom + link_dia + (bar_dia_comp / 2.0)
    
    # --- ULS BENDING (Cl. 6.1) ---
    lambda_pb, eta_pb = 0.8, 1.0  # For fck <= 50 MPa
    
    def uls_force_equilibrium(x):
        Fc = eta_pb * fcd * b * (lambda_pb * x)
        eps_s = 0.0035 * (d - x) / x
        Fs_tens = As_tens * min(eps_s * Es, fyd)
        Fs_comp = As_comp * max(min(0.0035 * (x - d_comp) / x * Es, fyd), -fyd) if As_comp > 0 else 0.0
        return Fc + Fs_comp - Fs_tens

    try:
        x_uls = brentq(uls_force_equilibrium, 1.0, d)
    except ValueError:
        x_uls = d

    eps_s_uls = 0.0035 * (d - x_uls) / x_uls
    fs_tens_uls = min(eps_s_uls * Es, fyd)
    Fc_uls = eta_pb * fcd * b * (lambda_pb * x_uls)
    
    if As_comp > 0:
        eps_sc_uls = 0.0035 * (x_uls - d_comp) / x_uls
        fs_comp_uls = max(min(eps_sc_uls * Es, fyd), -fyd)
    else:
        eps_sc_uls, fs_comp_uls = 0.0, 0.0
        
    M_rd = (Fc_uls * (d - 0.5 * lambda_pb * x_uls) + As_comp * fs_comp_uls * (d - d_comp)) / 1e6
    uls_bending_pass = M_rd >= M_ed

    # K-factor check (Simplified rectangular stress block)
    K = (M_ed * 1e6) / (b * (d**2) * fck)
    K_prime = 0.168 # Redistribution ratio delta = 1.0

    # --- ULS SHEAR (Cl. 6.2 & Irish NA) ---
    k_shear = min(1.0 + math.sqrt(200.0 / d), 2.0)
    rho_l = min(As_tens / (b * d), 0.02)
    CRd_c = 0.18 / gamma_c
    v_min = 0.035 * (k_shear**1.5) * (fck**0.5)
    VRd_c = max((CRd_c * k_shear * ((100 * rho_l * fck)**(1/3))) * b * d, v_min * b * d) / 1e3
    
    cot_theta = 2.5  # theta = 21.8 deg
    tan_theta = 1.0 / cot_theta
    z = 0.9 * d
    Asw = num_legs * (math.pi * link_dia**2 / 4.0)
    VRd_s = (Asw / link_spacing) * z * fyd * cot_theta / 1e3
    
    nu1 = 0.6 * (1.0 - fck / 250.0)
    alpha_cw = 1.0
    VRd_max = (alpha_cw * b * z * nu1 * fcd / (cot_theta + tan_theta)) / 1e3
    VRd = min(VRd_s, VRd_max)
    uls_shear_pass = VRd >= V_ed

    # --- SLS STRESS LIMITS (Cl. 7.2) ---
    phi_eff = 2.0  # Creep coefficient
    E_eff = Ecm / (1.0 + phi_eff)
    alpha_e = Es / E_eff
    
    A_a = 0.5 * b
    B_a = alpha_e * (As_tens + As_comp)
    C_a = -alpha_e * (As_tens * d + As_comp * d_comp)
    x_sls = (-B_a + math.sqrt(B_a**2 - 4 * A_a * C_a)) / (2 * A_a)
    
    I_cr = (b * (x_sls**3) / 3.0) + alpha_e * As_tens * ((d - x_sls)**2) + alpha_e * As_comp * ((x_sls - d_comp)**2)
    
    sigma_c_char = (M_char * 1e6 / I_cr) * x_sls
    sigma_s_char = alpha_e * (M_char * 1e6 / I_cr) * (d - x_sls)
    sigma_c_qp = (M_qp * 1e6 / I_cr) * x_sls
    sigma_s_qp = alpha_e * (M_qp * 1e6 / I_cr) * (d - x_sls)
    
    lim_c_char = 0.60 * fck
    lim_s_char = 0.80 * fyk
    lim_c_qp = 0.45 * fck

    # --- SLS CRACK WIDTHS (Cl. 7.3.4) ---
    h_c_eff = min(2.5 * (h - d), (h - x_sls) / 3.0, h / 2.0)
    Ac_eff = b * h_c_eff
    rho_p_eff = As_tens / Ac_eff
    
    k1, k2, k3, k4, kt = 0.8, 0.5, 3.4, 0.425, 0.4
    S_rmax = k3 * c_nom + (k1 * k2 * k4 * bar_dia_tens) / rho_p_eff
    
    eps_diff_calc = (sigma_s_qp - kt * (fctm / rho_p_eff) * (1.0 + alpha_e * rho_p_eff)) / Es
    eps_diff_min = 0.6 * (sigma_s_qp / Es)
    eps_diff = max(eps_diff_calc, eps_diff_min)
    w_k = S_rmax * eps_diff
    pass_crack = w_k <= w_max_req

    return locals()

# =============================================================================
# 2. STREAMLIT INTERFACE
# =============================================================================
st.title("Eurocode 2 (IS EN 1992-1-1 / IS EN 1992-2) Design Dashboard")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Cross Section & Cover")
    b = st.number_input("Width b (mm)", 150, 1500, 300, 25)
    h = st.number_input("Overall Depth h (mm)", 200, 2500, 600, 25)
    c_nom = st.number_input("Nominal Cover c_nom (mm)", 20, 100, 40, 5)
    
    st.header("2. Reinforcement Setup")
    st.subheader("Tension Rebar (Bottom)")
    num_bars_tens = st.number_input("Number of Tension Bars", 2, 12, 4)
    bar_dia_tens = st.selectbox("Bar Diameter (mm)", [12.0, 16.0, 20.0, 25.0, 32.0, 40.0], index=3)
    
    st.subheader("Compression Rebar (Top)")
    num_bars_comp = st.number_input("Number of Compression Bars", 0, 12, 2)
    bar_dia_comp = st.selectbox("Comp. Bar Diameter (mm)", [12.0, 16.0, 20.0, 25.0, 32.0], index=1)
    
    st.subheader("Shear Links")
    link_dia = st.selectbox("Link Diameter (mm)", [8.0, 10.0, 12.0, 16.0], index=1)
    link_spacing = st.number_input("Spacing s (mm)", 50, 500, 150, 25)
    num_legs = st.number_input("Shear Link Legs", 2, 6, 2, 2)
    
    st.header("3. Material Strengths")
    fck = st.number_input("Concrete fck (N/mm²)", 20.0, 90.0, 30.0, 5.0)
    fyk = st.number_input("Steel fyk (N/mm²)", 400.0, 600.0, 500.0, 50.0)
    Es = 200000.0
    
    st.header("4. Actions (Internal Forces)")
    M_ed = st.number_input("ULS Bending Moment M_Ed (kNm)", 0.0, 3000.0, 350.0, 10.0)
    V_ed = st.number_input("ULS Shear Force V_Ed (kN)", 0.0, 1500.0, 180.0, 10.0)
    M_char = st.number_input("SLS Characteristic Moment M_char (kNm)", 0.0, 3000.0, 220.0, 10.0)
    M_qp = st.number_input("SLS Quasi-Permanent Moment M_qp (kNm)", 0.0, 3000.0, 140.0, 10.0)
    w_max_req = st.number_input("Max Allowable Crack Width w_max (mm)", 0.10, 0.40, 0.30, 0.05)

# Execute Engine
r = analyze_section(b, h, c_nom, bar_dia_tens, num_bars_tens, bar_dia_comp, num_bars_comp, 
                    link_dia, link_spacing, num_legs, fck, fyk, Es, M_ed, V_ed, 
                    M_char, M_char*0.8, M_qp, w_max_req)

# Top Summary Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("ULS Moment Capacity (MRd)", f"{r['M_rd']:.1f} kNm", f"Target: {M_ed} kNm", delta_color="normal" if r['uls_bending_pass'] else "inverse")
m2.metric("ULS Shear Capacity (VRd)", f"{r['VRd']:.1f} kN", f"Target: {V_ed} kN", delta_color="normal" if r['uls_shear_pass'] else "inverse")
m3.metric("SLS Crack Width (wk)", f"{r['w_k']:.3f} mm", f"Limit: {w_max_req} mm", delta_color="normal" if r['pass_crack'] else "inverse")
m4.metric("SLS Comp. Stress (σc,qp)", f"{r['sigma_c_qp']:.1f} MPa", f"Limit: {r['lim_c_qp']:.1f} MPa", delta_color="normal" if r['sigma_c_qp']<=r['lim_c_qp'] else "inverse")

st.divider()

col_sketch, col_calc = st.columns([1, 1.8])

# --- CROSS SECTION DRAWING ---
with col_sketch:
    st.subheader("Cross Section Sketch")
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(-60, b + 60)
    ax.set_ylim(-60, h + 60)
    
    # Concrete core
    ax.add_patch(patches.Rectangle((0, 0), b, h, fill=True, color='#E8ECEF', ec='black', lw=2))
    
    # Shear Link
    link_w, link_h = b - 2*c_nom, h - 2*c_nom
    ax.add_patch(patches.Rectangle((c_nom, c_nom), link_w, link_h, fill=False, ec='#D9534F', lw=2, ls='--'))
    
    # Bottom Tension Rebar
    if num_bars_tens > 0:
        sp_t = (link_w - bar_dia_tens) / (num_bars_tens - 1) if num_bars_tens > 1 else 0
        x_t = c_nom + link_dia + bar_dia_tens/2
        y_t = c_nom + link_dia + bar_dia_tens/2
        for i in range(int(num_bars_tens)):
            ax.add_patch(patches.Circle((x_t + i*sp_t, y_t), bar_dia_tens/2, color='#292B2C'))
            
    # Top Compression Rebar
    if num_bars_comp > 0:
        sp_c = (link_w - bar_dia_comp) / (num_bars_comp - 1) if num_bars_comp > 1 else 0
        x_c = c_nom + link_dia + bar_dia_comp/2
        y_c = h - (c_nom + link_dia + bar_dia_comp/2)
        for i in range(int(num_bars_comp)):
            ax.add_patch(patches.Circle((x_c + i*sp_c, y_c), bar_dia_comp/2, color='#292B2C'))
            
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)

# --- DETAILED CALCULATION REPORT WITH FORMULAE & CLAUSES ---
with col_calc:
    st.subheader("Detailed Eurocode Calculation Sheet")
    
    tab1, tab2, tab3, tab4 = st.tabs(["1. Materials & Geoms", "2. ULS Bending", "3. ULS Shear", "4. SLS Stresses & Cracks"])
    
    # --- TAB 1: MATERIALS & GEOMETRY ---
    with tab1:
        st.markdown("### Material Strengths & Geometry Parameters")
        st.markdown("**IS EN 1992-1-1 Cl. 3.1 & Irish NA (Table 3.1):**")
        st.latex(r"f_{cd} = \alpha_{cc} \cdot \frac{f_{ck}}{\gamma_c} = 0.85 \cdot \frac{" + f"{fck:.1f}" + r"}{1.5} = " + f"{r['fcd']:.2f}" + r"\text{ MPa}")
        st.latex(r"f_{yd} = \frac{f_{yk}}{\gamma_s} = \frac{" + f"{fyk:.1f}" + r"}{1.15} = " + f"{r['fyd']:.2f}" + r"\text{ MPa}")
        st.latex(r"f_{ctm} = 0.30 \cdot f_{ck}^{2/3} = 0.30 \cdot (" + f"{fck:.1f}" + r")^{2/3} = " + f"{r['fctm']:.2f}" + r"\text{ MPa}")
        st.latex(r"E_{cm} = 22000 \cdot \left(\frac{f_{ck}+8}{10}\right)^{0.3} = " + f"{r['Ecm']:.0f}" + r"\text{ MPa}")
        
        st.markdown("**Reinforcement Areas & Effective Depths:**")
        st.latex(r"A_{s} = " + f"{num_bars_tens} \cdot \frac{{\pi \cdot {bar_dia_tens:.0f}^2}}{{4}} = {r['As_tens']:.0f}" + r"\text{ mm}^2")
        st.latex(r"d = h - c_{nom} - \phi_{link} - \frac{\phi_{bar}}{2} = " + f"{h} - {c_nom} - {link_dia} - {bar_dia_tens/2:.1f} = {r['d']:.1f}" + r"\text{ mm}")
        if num_bars_comp > 0:
            st.latex(r"A_{s2} = " + f"{num_bars_comp} \cdot \frac{{\pi \cdot {bar_dia_comp:.0f}^2}}{{4}} = {r['As_comp']:.0f}" + r"\text{ mm}^2")
            st.latex(r"d' = c_{nom} + \phi_{link} + \frac{\phi_{bar,c}}{2} = " + f"{r['d_comp']:.1f}" + r"\text{ mm}")

    # --- TAB 2: ULS BENDING ---
    with tab2:
        st.markdown("### Ultimate Bending Resistance (IS EN 1992-1-1 Cl. 6.1)")
        st.markdown("**Normalized Bending Moment ($K$):**")
        st.latex(r"K = \frac{M_{Ed}}{b \cdot d^2 \cdot f_{ck}} = \frac{" + f"{M_ed*1e6:.0f}" + r"}{" + f"{b} \cdot {r['d']:.1f}^2 \cdot {fck}" + r"} = " + f"{r['K']:.3f}")
        
        if r['K'] <= r['K_prime']:
            st.write(f"Since $K = {r['K']:.3f} \le K' = 0.168$, singly reinforced section assumptions are valid without requiring extra compression steel.")
        else:
            st.warning(f"Since $K = {r['K']:.3f} > K' = 0.168$, compression steel is required to limit neutral axis depth.")

        st.markdown("**Neutral Axis Depth ($x$) & Forces Equilibrium:**")
        st.latex(r"F_c = \eta \cdot f_{cd} \cdot b \cdot (\lambda \cdot x) = 1.0 \cdot " + f"{r['fcd']:.2f} \cdot {b} \cdot (0.8 \cdot x)")
        st.latex(r"x_{uls} = " + f"{r['x_uls']:.1f}" + r"\text{ mm} \quad \implies \quad \frac{x}{d} = " + f"{r['x_uls']/r['d']:.3f}" + r" \le 0.45 \text{ (Cl. 5.6.3)}")
        
        st.markdown("**Bending Resistance Moment ($M_{Rd}$):**")
        st.latex(r"M_{Rd} = F_c \cdot \left(d - 0.5 \cdot \lambda \cdot x\right) + F_{sc} \cdot (d - d') = " + f"{r['M_rd']:.2f}" + r"\text{ kNm}")
        
        if r['uls_bending_pass']:
            st.success(f"**PASS:** M_Rd = {r['M_rd']:.2f} kNm >= M_Ed = {M_ed:.2f} kNm")
        else:
            st.error(f"**FAIL:** M_Rd = {r['M_rd']:.2f} kNm < M_Ed = {M_ed:.2f} kNm")

    # --- TAB 3: ULS SHEAR ---
    with tab3:
        st.markdown("### Ultimate Shear Resistance (IS EN 1992-1-1 Cl. 6.2)")
        st.markdown("**1. Concrete Shear Capacity without Links ($V_{Rd,c}$ - Cl. 6.2.2):**")
        st.latex(r"k = 1 + \sqrt{\frac{200}{d}} = 1 + \sqrt{\frac{200}{" + f"{r['d']:.1f}" + r"}} = " + f"{r['k_shear']:.3f}" + r" \le 2.0")
        st.latex(r"\rho_l = \frac{A_{sl}}{b \cdot d} = \frac{" + f"{r['As_tens']:.0f}" + r"}{" + f"{b} \cdot {r['d']:.1f}" + r"} = " + f"{r['rho_l']:.4f}" + r" \le 0.02")
        st.latex(r"V_{Rd,c} = \left[ C_{Rd,c} \cdot k \cdot \left(100 \cdot \rho_l \cdot f_{ck}\right)^{1/3} \right] b \cdot d = " + f"{r['VRd_c']:.2f}" + r"\text{ kN}")

        st.markdown("**2. Shear Link Capacity ($V_{Rd,s}$ - Cl. 6.2.3 using $\cot\theta = 2.5$):**")
        st.latex(r"z = 0.9 \cdot d = 0.9 \cdot " + f"{r['d']:.1f} = {r['z']:.1f}" + r"\text{ mm}")
        st.latex(r"V_{Rd,s} = \frac{A_{sw}}{s} \cdot z \cdot f_{yd} \cdot \cot\theta = \frac{" + f"{r['Asw']:.1f}" + r"}{" + f"{link_spacing}" + r"} \cdot " + f"{r['z']:.1f} \cdot {r['fyd']:.1f} \cdot 2.5 = " + f"{r['VRd_s']:.2f}" + r"\text{ kN}")

        st.markdown("**3. Maximum Concrete Strut Capacity ($V_{Rd,max}$ - Eq 6.9):**")
        st.latex(r"\nu_1 = 0.6 \cdot \left(1 - \frac{f_{ck}}{250}\right) = " + f"{r['nu1']:.3f}")
        st.latex(r"V_{Rd,max} = \frac{\alpha_{cw} \cdot b \cdot z \cdot \nu_1 \cdot f_{cd}}{\cot\theta + \tan\theta} = " + f"{r['VRd_max']:.2f}" + r"\text{ kN}")

        st.latex(r"V_{Rd} = \min(V_{Rd,s}, V_{Rd,max}) = " + f"{r['VRd']:.2f}" + r"\text{ kN}")
        
        if r['uls_shear_pass']:
            st.success(f"**PASS:** V_Rd = {r['VRd']:.2f} kN >= V_Ed = {V_ed:.2f} kN")
        else:
            st.error(f"**FAIL:** V_Rd = {r['VRd']:.2f} kN < V_Ed = {V_ed:.2f} kN")

    # --- TAB 4: SLS STRESSES & CRACKING ---
    with tab4:
        st.markdown("### Serviceability Stress Limits & Crack Widths (Cl. 7.2 & 7.3)")
        
        st.markdown("**Modular Ratio & Cracked Section Neutral Axis ($x_{sls}$):**")
        st.latex(r"\alpha_e = \frac{E_s}{E_{eff}} = \frac{E_s}{E_{cm}/(1+\varphi_{eff})} = \frac{200000}{" + f"{r['E_eff']:.1f}" + r"} = " + f"{r['alpha_e']:.2f}")
        st.latex(r"x_{sls} = " + f"{r['x_sls']:.1f}" + r"\text{ mm} \quad \implies \quad I_{cr} = " + f"{r['I_cr']/1e6:.1f} \times 10^6" + r"\text{ mm}^4")

        st.markdown("**1. Stress Limit Checks (Cl. 7.2):**")
        st.latex(r"\sigma_{c,char} = \frac{M_{char}}{I_{cr}} \cdot x_{sls} = " + f"{r['sigma_c_char']:.2f}" + r"\text{ MPa } \le 0.60 f_{ck} = " + f"{r['lim_c_char']:.1f}" + r"\text{ MPa}")
        st.latex(r"\sigma_{s,char} = \alpha_e \cdot \frac{M_{char}}{I_{cr}} \cdot (d - x_{sls}) = " + f"{r['sigma_s_char']:.2f}" + r"\text{ MPa } \le 0.80 f_{yk} = " + f"{r['lim_s_char']:.1f}" + r"\text{ MPa}")
        st.latex(r"\sigma_{c,qp} = \frac{M_{qp}}{I_{cr}} \cdot x_{sls} = " + f"{r['sigma_c_qp']:.2f}" + r"\text{ MPa } \le 0.45 f_{ck} = " + f"{r['lim_c_qp']:.1f}" + r"\text{ MPa}")

        st.markdown("**2. Crack Width Calculation (Cl. 7.3.4 under $M_{qp}$):**")
        st.latex(r"h_{c,eff} = \min\left[2.5(h-d), \frac{h-x}{3}, \frac{h}{2}\right] = " + f"{r['h_c_eff']:.1f}" + r"\text{ mm}")
        st.latex(r"\rho_{p,eff} = \frac{A_s}{b \cdot h_{c,eff}} = " + f"{r['rho_p_eff']:.4f}")
        st.latex(r"S_{r,max} = k_3 \cdot c_{nom} + \frac{k_1 k_2 k_4 \phi}{\rho_{p,eff}} = 3.4( " + f"{c_nom}" + r") + \frac{0.8 \cdot 0.5 \cdot 0.425 \cdot " + f"{bar_dia_tens}" + r"}{" + f"{r['rho_p_eff']:.4f}" + r"} = " + f"{r['S_rmax']:.1f}" + r"\text{ mm}")
        st.latex(r"(\varepsilon_{sm} - \varepsilon_{cm}) = \max\left[ \frac{\sigma_{s,qp} - k_t \frac{f_{ctm}}{\rho_{p,eff}}(1+\alpha_e \rho_{p,eff})}{E_s}, \, 0.6 \frac{\sigma_{s,qp}}{E_s} \right] = " + f"{r['eps_diff']*1e3:.3f} \times 10^{-3}")
        st.latex(r"w_k = S_{r,max} \cdot (\varepsilon_{sm} - \varepsilon_{cm}) = " + f"{r['S_rmax']:.1f}" + r"\cdot " + f"{r['eps_diff']:.6f}" + r" = " + f"{r['w_k']:.3f}" + r"\text{ mm}")

        if r['pass_crack']:
            st.success(f"**PASS:** w_k = {r['w_k']:.3f} mm <= w_max = {w_max_req:.2f} mm")
        else:
            st.error(f"**FAIL:** w_k = {r['w_k']:.3f} mm > w_max = {w_max_req:.2f} mm")
