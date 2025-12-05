import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests  # NEW: for HTTP call to Lambda

# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Cantilever Beam Designer",
    layout="wide"
)

st.title("📐 Cantilever Beam Design Tool")

# ===============================
# BACKEND CONFIG (for Streamlit Cloud)
# ===============================
# backend_mode: "lambda_http" (default) or "dummy"
BACKEND_MODE = st.secrets.get("backend_mode", "lambda_http")
LAMBDA_URL = st.secrets.get("lambda_url", "")

st.sidebar.caption(f"Backend mode: {BACKEND_MODE}")

#=====================================================
def draw_cantilever_load(L, q0, b, h):
    """
    Draw cantilever beam with snow-load-shaped distributed load.

    The arrow envelope follows |q(x)| with
        q(x) = q0 * (x^2 / L^2 - 1),
    so |q(x)| is largest at x = 0 and goes smoothly to zero at x = L.
    Arrows are drawn from the blue curve DOWN to the beam.
    """
    xs = np.linspace(0.0, L, 21)

    shape = np.maximum(0.0, 1.0 - (xs / L) ** 2)
    q_mag = np.abs(q0) * shape

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    beam_y = -0.10
    ax.plot([0, L], [beam_y, beam_y], color="black", linewidth=4)

    ax.plot([0, 0], [beam_y - 0.15, beam_y + 0.15], color="black", linewidth=6)
    ax.text(-0.07, beam_y + 0.13, "Fixed support", fontsize=14, ha="right")

    scale_per_unit = 2.0e-4
    max_arrow_cap  = 1.6

    arrow_heights = np.minimum(scale_per_unit * q_mag, max_arrow_cap)

    x_dense = np.linspace(0.0, L, 200)
    shape_dense = np.maximum(0.0, 1.0 - (x_dense / L) ** 2)
    q_dense = np.abs(q0) * shape_dense
    h_dense = np.minimum(scale_per_unit * q_dense, max_arrow_cap)

    ax.plot(x_dense, beam_y + h_dense, color="tab:blue", linewidth=1.6)

    for xi, ah in zip(xs, arrow_heights):
        if ah <= 0.05:
            continue

        tail_y = beam_y + ah
        tip_y  = beam_y

        ax.arrow(
            xi,
            tail_y,
            0.0,
            tip_y - tail_y,
            length_includes_head=True,
            head_width=0.007 * L,
            head_length=0.05 * L,
            linewidth=1.6,
            color="tab:blue"
        )

    ax.text(
        0.4 * L,
        beam_y + max_arrow_cap * 0.75,
        r"$q(x) = q_0\!\left(\dfrac{x^2}{L^2} - 1\right)$",
        fontsize=18,
        ha="center",
        va="bottom"
    )

    ax.set_ylim(-0.55, max_arrow_cap * 2.0)
    ax.set_xlim(-0.05, L + 0.8)
    ax.axis("off")

    st.pyplot(fig)


# ===============================
# CLOUD-BACKED BEAM FIELDS (Lambda via HTTP)
# ===============================
def beam_fields_lambda_http(E, I, L, q0, N=200):
    """
    Call AWS Lambda over HTTPS to evaluate the beam response,
    then reconstruct q(x) locally.

    Expects the HTTP endpoint to return JSON directly with:
      {"x": [...], "w": [...], "M": [...], "V": [...]}
    """
    if not LAMBDA_URL:
        st.error("Lambda URL not configured. Set st.secrets['lambda_url'].")
        st.stop()

    event = {
        "params": {
            "L": float(L),
            "E": float(E),
            "I": float(I),
            "q0": float(q0),
            "n_points": int(N),
        }
    }

    try:
        resp = requests.post(LAMBDA_URL, json=event, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        st.error(f"Error calling Lambda endpoint: {exc}")
        st.stop()

    # NEW: expect plain x,w,M,V JSON from the proxy
    try:
        body = resp.json()
    except Exception as exc:
        st.error(f"Could not parse JSON from Lambda HTTP response: {exc}")
        st.text(resp.text)
        st.stop()

    try:
        x = np.array(body["x"])
        w = np.array(body["w"])
        M = np.array(body["M"])
        V = np.array(body["V"])
    except KeyError as exc:
        st.error(f"Missing key in Lambda response: {exc}")
        st.json(body)
        st.stop()

    xi = x / L
    q = q0 * (xi**2 - 1.0)

    return x, w, M, V, q


def beam_fields_dummy(E, I, L, q0, N=200):
    """
    Dummy local backend (for development if Lambda is not wired yet).
    Produces simple Euler-Bernoulli-style shapes, not your PINN.
    """
    x = np.linspace(0.0, L, N)
    xi = x / L

    # super crude shapes just to have something:
    w = -q0 * xi**2 * (3 - 2*xi) * 1e-5
    M = -q0 * L * xi * (1 - xi)
    V = q0 * (1 - 2*xi)
    q  = q0 * (xi**2 - 1.0)

    return x, w, M, V, q


def beam_fields(E, I, L, q0, N=200):
    """
    Unified entry point for computing beam fields.
    """
    if BACKEND_MODE == "lambda_http":
        return beam_fields_lambda_http(E, I, L, q0, N)
    else:
        # Fallback for local/dummy mode
        return beam_fields_dummy(E, I, L, q0, N)


# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("Beam & Load Parameters")

rho_steel = 7850        # kg/m^3
g = 9.81

st.sidebar.markdown("### Global inputs")

# 👉 E in GPa, sigma_y in MPa
E_GPa_str       = st.sidebar.text_input("Young's modulus E (GPa)", value="200.0")
sigma_y_MPa_str = st.sidebar.text_input("Yield stress σᵧ (MPa)",   value="250.0")

# Beam length and load
L_str  = st.sidebar.text_input("Beam length L (m)", value="2.0")
q0_str = st.sidebar.text_input("Load magnitude q₀ (N/m)", value="2000.0")

st.sidebar.markdown("### Rectangular section (enter in cm)")

b_cm_str = st.sidebar.text_input("Width b (cm)",  value="5.0")
h_cm_str = st.sidebar.text_input("Height h (cm)", value="10.0")


def parse_float(label, s, default):
    try:
        return float(s), None
    except ValueError:
        return default, f"Could not parse **{label}** = '{s}'. Using default {default}."


# Parse all inputs with defaults (in user units)
E_GPa_val,       err_E       = parse_float("E (GPa)",     E_GPa_str,       200.0)
sigma_y_MPa_val, err_sigma_y = parse_float("σᵧ (MPa)",    sigma_y_MPa_str, 250.0)
L_val,           err_L       = parse_float("L (m)",       L_str,           2.0)
q0_val,          err_q0      = parse_float("q₀ (N/m)",    q0_str,          2000.0)
b_cm,            err_b       = parse_float("b (cm)",      b_cm_str,        5.0)
h_cm,            err_h       = parse_float("h (cm)",      h_cm_str,        10.0)

for err in [err_E, err_sigma_y, err_L, err_q0, err_b, err_h]:
    if err is not None:
        st.sidebar.warning(err)

# Convert to SI units
E       = E_GPa_val * 1e9        # GPa → Pa
sigma_y = sigma_y_MPa_val * 1e6  # MPa → Pa
L       = L_val
q0      = q0_val
b       = b_cm / 100.0
h       = h_cm / 100.0

# Section properties
I = b * h**3 / 12.0
A = b * h

draw_cantilever_load(L, q0, b, h)

st.markdown(
    r"""
This tool lets you explore how a cantilever beam responds to a snow-load-shaped
distributed load,
$$ 
q(x) = q_0\left(\dfrac{x^2}{L^2} - 1\right). 
$$

The beam has Young's modulus $E$ (entered in GPa) and yield stress $\sigma_y$
(entered in MPa); both are converted internally to Pascals. Your goal is to
design a **rectangular steel beam** with width $b$ and height $h$ that is strong
enough while keeping the cross-sectional area $A = b h$ as small as possible.
The required bending safety factor is
$$ 
SF = \frac{\sigma_y}{\sigma_{\max}} \ge 3, 
$$
where $\sigma_{\max}$ occurs at the fixed end of the beam.

You can adjust $E$, $\sigma_y$, $L$, $q_0$, $b$, and $h$, and the app will
instantly update the deflection $w(x)$, moment $M(x)$, shear $V(x)$, and bending
stress. These results come from a **Physics-Informed Neural Network (PINN)**
deployed in the cloud via AWS Lambda.
""",
)

st.sidebar.markdown("### Derived section properties")
st.sidebar.write(f"**Moment of Inertia:** {I:.3e} m⁴")
st.sidebar.write(f"**Area:** {A:.3e} m²")
st.sidebar.write(f"**b:** {b*1000:.1f} mm, **h:** {h*1000:.1f} mm")

# ===============================
# COMPUTE BEAM RESPONSE (via unified backend)
# ===============================
x, w, M, V, q = beam_fields(E, I, L, q0)

# Maximum tensile stress at the fixed end (top fiber)
M_max = np.min(M)   # most negative = tension at top
sigma_max = abs(M_max) * (h/2) / I
SF = sigma_y / sigma_max

weight_per_m = rho_steel * A * g

# ===============================
# PLOTS
# ===============================
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(x, w)
    ax.set_title("Deflection w(x)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("w (m)")
    ax.grid(True)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.plot(x, M)
    ax2.set_title("Moment M(x)")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("M (N·m)")
    ax2.grid(True)
    st.pyplot(fig2)

with col2:
    fig3, ax3 = plt.subplots(figsize=(5,3))
    ax3.plot(x, V)
    ax3.set_title("Shear V(x)")
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("V (N)")
    ax3.grid(True)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(x, q)
    ax4.set_title("Distributed Load q(x)")
    ax4.set_xlabel("x (m)"); ax4.set_ylabel("q (N/m)")
    ax4.grid(True)
    st.pyplot(fig4)

# ===============================
# DESIGN METRICS
# ===============================
st.header("Design Evaluation")

st.write(f"**Max Moment Mₘₐₓ:** {M_max:.3e} N·m")
st.write(f"**Max Stress σₘₐₓ:** {sigma_max:.3e} Pa")
st.write(f"**Yield Stress σᵧ:** {sigma_y:.3e} Pa")
st.write(f"**Safety Factor SF:** {SF:.2f}")
st.write(f"**Weight per meter:** {weight_per_m:.2f} N/m")

if SF < 3:
    st.error("⚠️ Safety factor below 3. Increase height h or choose a stronger section.")
elif SF > 3.5:
    st.warning(f"Beam is overdesigned by {(SF/3 - 1)*100:.1f}%. You can reduce height.")
else:
    st.success("✔️ Beam meets the safety factor requirement (SF ≈ 3). Optimal design!")