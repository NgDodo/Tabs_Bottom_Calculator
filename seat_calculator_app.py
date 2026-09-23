"""
seat_calculator_app.py -- interactive Streamlit UI for the seat bottom
panel and tab calculator.

Reuses the calculation functions from seat_calculator.py (same formulas
as the CLI and the Notion database) so there is one source of truth.

Presets are stored in presets.json next to this file, and are editable
from the UI -- edit the "Edit / save presets" expander in either tab,
save over the current preset or save as a new one.

Run with:
    streamlit run seat_calculator_app.py
"""

import json
import math
import os
from dataclasses import asdict

import streamlit as st
import plotly.graph_objects as go

from seat_calculator import (
    PanelInputs, panel_calc,
    PanelSolidInputs, panel_solid_calc,
    TabInputs, tabs_calc,
)

# ---------------------------------------------------------------------------
# Real seat bottom geometry (from CAD) -- now just the initial widget
# defaults, since front/back width and length are editable inputs.
# ---------------------------------------------------------------------------
FRONT_MM = 340.75
BACK_MM = 432.97
PANEL_LENGTH_MM = 495.15

PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"

# ---------------------------------------------------------------------------
# Presets -- persisted to presets.json, editable from the UI.
# Values below are PLACEHOLDERS, swap for real datasheet/coupon data.
# ---------------------------------------------------------------------------
PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")

PANEL_FIELDS = ["t_f_mm", "E_f_MPa", "t_c_mm", "G_c_MPa", "E_c_MPa",
                "sigma_allow_MPa", "tau_allow_MPa"]
PANEL_SOLID_FIELDS = ["E_eff_MPa", "t_total_mm", "sigma_allow_MPa"]
TAB_FIELDS = ["E_MPa", "sigma_yield_MPa", "tau_weld_allow_MPa"]

DEFAULT_PANEL_PRESETS = {
    "Custom": dict(t_f_mm=0.55, E_f_MPa=55000.0, t_c_mm=6.35, G_c_MPa=22.0,
                   E_c_MPa=3000.0, sigma_allow_MPa=500.0, tau_allow_MPa=0.9),
    "CF twill skins + Nomex honeycomb (~3 pcf)": dict(
        t_f_mm=0.55, E_f_MPa=55000.0, t_c_mm=6.35, G_c_MPa=22.0,
        E_c_MPa=3000.0, sigma_allow_MPa=500.0, tau_allow_MPa=0.9),
    "CF twill skins + Aluminum honeycomb (~3.1 pcf)": dict(
        t_f_mm=0.55, E_f_MPa=55000.0, t_c_mm=6.35, G_c_MPa=140.0,
        E_c_MPa=8000.0, sigma_allow_MPa=500.0, tau_allow_MPa=1.7),
}

DEFAULT_PANEL_SOLID_PRESETS = {
    "Custom": dict(E_eff_MPa=20000.0, t_total_mm=7.5, sigma_allow_MPa=300.0),
    "6061-T6 Aluminum plate": dict(E_eff_MPa=69000.0, t_total_mm=6.35, sigma_allow_MPa=240.0),
    "Solid CF laminate (quasi-isotropic)": dict(E_eff_MPa=45000.0, t_total_mm=4.0, sigma_allow_MPa=400.0),
}

DEFAULT_TAB_PRESETS = {
    "Custom": dict(E_MPa=205000.0, sigma_yield_MPa=460.0, tau_weld_allow_MPa=150.0),
    "AISI 4130 Steel": dict(E_MPa=205000.0, sigma_yield_MPa=460.0, tau_weld_allow_MPa=150.0),
    "6061-T6 Aluminum": dict(E_MPa=69000.0, sigma_yield_MPa=240.0, tau_weld_allow_MPa=90.0),
}


def load_presets():
    if os.path.exists(PRESETS_PATH):
        try:
            with open(PRESETS_PATH) as f:
                data = json.load(f)
            return (data.get("panel_presets", DEFAULT_PANEL_PRESETS),
                    data.get("panel_solid_presets", DEFAULT_PANEL_SOLID_PRESETS),
                    data.get("tab_presets", DEFAULT_TAB_PRESETS))
        except Exception:
            pass
    return ({k: dict(v) for k, v in DEFAULT_PANEL_PRESETS.items()},
            {k: dict(v) for k, v in DEFAULT_PANEL_SOLID_PRESETS.items()},
            {k: dict(v) for k, v in DEFAULT_TAB_PRESETS.items()})


def save_presets():
    with open(PRESETS_PATH, "w") as f:
        json.dump({"panel_presets": st.session_state.panel_presets,
                   "panel_solid_presets": st.session_state.panel_solid_presets,
                   "tab_presets": st.session_state.tab_presets}, f, indent=2)


if "panel_presets" not in st.session_state:
    panel_presets, panel_solid_presets, tab_presets = load_presets()
    st.session_state.panel_presets = panel_presets
    st.session_state.panel_solid_presets = panel_solid_presets
    st.session_state.tab_presets = tab_presets


# ---------------------------------------------------------------------------
# 3D visualization helpers
# ---------------------------------------------------------------------------

def make_hexahedron(bottom_pts, top_z, top_pts=None, color="#3498db", name=""):
    """bottom_pts: 4 (x, y) tuples at z=0, in order FL, FR, BR, BL.
    top_pts: same, at z=top_z (defaults to same x,y as bottom, i.e. a
    straight prism -- pass different points for a tapered/trapezoidal solid)."""
    if top_pts is None:
        top_pts = bottom_pts
    x = [p[0] for p in bottom_pts] + [p[0] for p in top_pts]
    y = [p[1] for p in bottom_pts] + [p[1] for p in top_pts]
    z = [0, 0, 0, 0, top_z, top_z, top_z, top_z]
    i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.85,
                      flatshading=True, name=name)


def panel_figure(front_mm, back_mm, length_mm, thickness, passed):
    hf, hb = front_mm / 2, back_mm / 2
    bottom = [(-hf, 0), (hf, 0), (hb, length_mm), (-hb, length_mm)]
    mesh = make_hexahedron(bottom, thickness, color=PASS_COLOR if passed else FAIL_COLOR,
                            name="panel")
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title="width (mm)", yaxis_title="length (mm)", zaxis_title="thickness (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0), height=420,
    )
    return fig


def hole_cylinder_mesh(cx, cy, radius, z0, z1, color="#111111", n=24, name="hole"):
    """A through-hole, rendered as an open tube (no end caps) so it reads
    visually as a hole rather than a solid rod."""
    xs, ys, zs = [], [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        xs += [x, x]
        ys += [y, y]
        zs += [z0, z1]
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        b0, t0 = 2 * i, 2 * i + 1
        b1, t1 = 2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1
        i_idx += [b0, b1]
        j_idx += [b1, t1]
        k_idx += [t0, t0]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i_idx, j=j_idx, k=k_idx, color=color, opacity=1.0,
                      flatshading=True, name=name)


def fillet_arc_mesh(y_center, z_center, radius, theta_start, theta_end, x0, x1,
                     color="#3498db", n=8, name="fillet"):
    """A rounded strip swept along x between two angles in the y-z plane --
    used to round off a sharp edge of the tab block."""
    thetas = [theta_start + (theta_end - theta_start) * i / (n - 1) for i in range(n)]
    arc_pts = [(y_center + radius * math.cos(th), z_center + radius * math.sin(th)) for th in thetas]
    xs, ys, zs = [], [], []
    for x in (x0, x1):
        for (y, z) in arc_pts:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 1):
        a0, a1 = i, i + 1
        b0, b1 = n + i, n + i + 1
        i_idx += [a0, b0]
        j_idx += [b0, b1]
        k_idx += [a1, a1]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i_idx, j=j_idx, k=k_idx, color=color, opacity=0.95,
                      flatshading=True, name=name)


def tab_figure(b_mm, t_mm, L_mm, passed, hole_diameter_mm=0.0, hole_position_mm=0.0,
               fillet_radius_mm=0.0):
    color = PASS_COLOR if passed else FAIL_COLOR
    bottom = [(0, 0), (b_mm, 0), (b_mm, L_mm), (0, L_mm)]
    traces = [make_hexahedron(bottom, t_mm, color=color, name="tab")]

    if hole_diameter_mm > 0:
        traces.append(hole_cylinder_mesh(b_mm / 2, hole_position_mm, hole_diameter_mm / 2,
                                          0, t_mm))

    if fillet_radius_mm > 0:
        # Clamp so the fillet can't exceed the tab's own thickness -- a radius
        # larger than t/2 would make the two arcs overlap and look broken.
        r = min(fillet_radius_mm, t_mm / 2 - 0.01)
        if r > 0:
            # Top-root edge: rounds the corner at (y=0, z=t)
            traces.append(fillet_arc_mesh(r, t_mm - r, r, math.pi / 2, math.pi,
                                           0, b_mm, color=color, name="fillet_top"))
            # Bottom-root edge: rounds the corner at (y=0, z=0)
            traces.append(fillet_arc_mesh(r, r, r, math.pi, 3 * math.pi / 2,
                                           0, b_mm, color=color, name="fillet_bottom"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="width b (mm)", yaxis_title="length L (mm)", zaxis_title="thickness t (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0), height=420, showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def margin_metrics(outputs, keys):
    cols = st.columns(len(keys))
    all_pass = True
    for col, key in zip(cols, keys):
        val = outputs[key]
        passed = val >= 0
        all_pass = all_pass and passed
        col.metric(key, f"{val:+.2f}", delta="PASS" if passed else "FAIL",
                   delta_color="normal" if passed else "inverse")
    return all_pass


def preset_editor(presets_key, select_key, edit_fields, key_prefix, apply_fn):
    """Lets the user edit the currently-selected preset's values and save
    them back to presets.json, either overwriting it or as a new preset."""
    presets = st.session_state[presets_key]
    current_name = st.session_state[select_key]

    def save_existing():
        edited = {f: st.session_state[f"{key_prefix}_edit_{f}"] for f in edit_fields}
        st.session_state[presets_key][current_name] = edited
        save_presets()
        apply_fn()

    def save_as_new():
        new_name = st.session_state.get(f"{key_prefix}_new_preset_name", "").strip()
        if not new_name:
            return
        edited = {f: st.session_state[f"{key_prefix}_edit_{f}"] for f in edit_fields}
        st.session_state[presets_key][new_name] = edited
        save_presets()
        st.session_state[select_key] = new_name
        apply_fn()

    with st.expander("Edit / save presets"):
        st.caption(f"Editing values for: **{current_name}**")
        for f in edit_fields:
            st.number_input(f, value=float(presets[current_name][f]),
                             key=f"{key_prefix}_edit_{f}")
        c1, c2 = st.columns(2)
        c1.button(f"Save over '{current_name}'", key=f"{key_prefix}_save_existing",
                  on_click=save_existing)
        c2.text_input("New preset name", key=f"{key_prefix}_new_preset_name",
                       label_visibility="collapsed", placeholder="New preset name")
        c2.button("Save as new preset", key=f"{key_prefix}_save_new", on_click=save_as_new)


def sweep_section(param_names, defaults, calc_fn, dataclass_type, key_prefix):
    with st.expander("Sweep a variable"):
        var = st.selectbox("Variable to sweep", param_names, key=f"{key_prefix}_sweep_var")
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Start", value=float(defaults[var]) * 0.5, key=f"{key_prefix}_start")
        stop = c2.number_input("Stop", value=float(defaults[var]) * 1.5, key=f"{key_prefix}_stop")
        steps = c3.number_input("Steps", value=20, min_value=2, step=1, key=f"{key_prefix}_steps")
        output_key = st.selectbox("Output to plot",
                                   list(calc_fn(dataclass_type(**defaults)).keys()),
                                   key=f"{key_prefix}_sweep_out")
        if st.button("Run sweep", key=f"{key_prefix}_run_sweep"):
            xs, ys = [], []
            step_size = (stop - start) / (steps - 1)
            for i in range(int(steps)):
                v = start + i * step_size
                kwargs = dict(defaults)
                kwargs[var] = v
                out = calc_fn(dataclass_type(**kwargs))
                xs.append(v)
                ys.append(out[output_key])
            fig = go.Figure(data=go.Scatter(x=xs, y=ys, mode="lines+markers"))
            if output_key.startswith("MS"):
                fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title=var, yaxis_title=output_key, height=350,
                               margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel tab
# ---------------------------------------------------------------------------

def apply_panel_preset():
    preset = st.session_state.panel_presets[st.session_state.panel_preset]
    for f in PANEL_FIELDS:
        st.session_state[f"panel_{f}"] = preset[f]
        st.session_state[f"panel_edit_{f}"] = preset[f]


def apply_panel_solid_preset():
    preset = st.session_state.panel_solid_presets[st.session_state.panel_solid_preset]
    for f in PANEL_SOLID_FIELDS:
        st.session_state[f"panelsolid_{f}"] = preset[f]
        st.session_state[f"panelsolid_edit_{f}"] = preset[f]


def render_panel_tab():
    left, right = st.columns([2, 1])

    with right:
        st.subheader("Construction")
        mode = st.radio("Panel type", ["Sandwich (facesheet + core)", "Solid"],
                         key="panel_mode", horizontal=True)
        is_sandwich = mode.startswith("Sandwich")

        st.subheader("Geometry")
        front_mm = st.number_input("Front width (mm)", value=FRONT_MM, key="panel_front_mm",
                                    help="3D model only -- not used in the calculation. Back "
                                         "width governs over front for every span-dependent "
                                         "check, so front width is shown for reference only.")
        back_mm = st.number_input("Back width (mm)", value=BACK_MM, key="panel_back_mm",
                                   help="Used directly as the calculation span -- this is the "
                                        "only geometry input that actually feeds the math.")
        length_mm = st.number_input("Length, front-to-back (mm)", value=PANEL_LENGTH_MM,
                                     key="panel_length_mm",
                                     help="3D model only -- not used in the calculation. The "
                                          "beam-strip model only checks a single cross-car "
                                          "span, not the fore-aft direction.")

        st.subheader("Load")
        P_N = st.number_input("Design load P (N)", value=3560.0, key="panel_P_N")
        w_mm = st.number_input("Loaded width w (mm)", value=100.0, key="panel_w_mm",
                                help="Contact patch width (e.g. a shoe), not the panel's "
                                     "physical width.")

        if is_sandwich:
            st.subheader("Material preset")
            st.selectbox("Panel construction", list(st.session_state.panel_presets),
                         key="panel_preset", on_change=apply_panel_preset)
            preset = st.session_state.panel_presets[st.session_state.panel_preset]

            st.subheader("Variable inputs")
            t_f_mm = st.number_input("Facesheet (skin) thickness t_f (mm)",
                                      value=preset["t_f_mm"], key="panel_t_f_mm")
            E_f_MPa = st.number_input("Facesheet modulus E_f (MPa)",
                                       value=preset["E_f_MPa"], key="panel_E_f_MPa")
            t_c_mm = st.number_input("Core thickness t_c (mm)",
                                      value=preset["t_c_mm"], key="panel_t_c_mm")
            G_c_MPa = st.number_input("Core shear modulus G_c (MPa)",
                                       value=preset["G_c_MPa"], key="panel_G_c_MPa")
            E_c_MPa = st.number_input("Core modulus E_c (MPa, wrinkling only)",
                                       value=preset["E_c_MPa"], key="panel_E_c_MPa")
            sigma_allow_MPa = st.number_input("Facesheet allowable stress (MPa)",
                                               value=preset["sigma_allow_MPa"], key="panel_sigma_allow_MPa")
            tau_allow_MPa = st.number_input("Core allowable shear stress (MPa)",
                                             value=preset["tau_allow_MPa"], key="panel_tau_allow_MPa")

            preset_editor("panel_presets", "panel_preset", PANEL_FIELDS, "panel", apply_panel_preset)

            inputs = PanelInputs(P_N=P_N, L_mm=back_mm, w_mm=w_mm, t_f_mm=t_f_mm, E_f_MPa=E_f_MPa,
                                  t_c_mm=t_c_mm, G_c_MPa=G_c_MPa, E_c_MPa=E_c_MPa,
                                  sigma_allow_MPa=sigma_allow_MPa, tau_allow_MPa=tau_allow_MPa)
            outputs = panel_calc(inputs)
            passed = all(outputs[k] >= 0 for k in ("MS_face", "MS_shear", "MS_wrinkle"))
            thickness = 2 * t_f_mm + t_c_mm
        else:
            st.subheader("Material preset")
            st.selectbox("Panel material", list(st.session_state.panel_solid_presets),
                         key="panel_solid_preset", on_change=apply_panel_solid_preset)
            preset = st.session_state.panel_solid_presets[st.session_state.panel_solid_preset]

            st.subheader("Variable inputs")
            E_eff_MPa = st.number_input("Modulus E (MPa)", value=preset["E_eff_MPa"],
                                         key="panelsolid_E_eff_MPa")
            t_total_mm = st.number_input("Thickness t (mm)", value=preset["t_total_mm"],
                                          key="panelsolid_t_total_mm")
            sigma_allow_MPa = st.number_input("Allowable stress (MPa)", value=preset["sigma_allow_MPa"],
                                               key="panelsolid_sigma_allow_MPa")

            preset_editor("panel_solid_presets", "panel_solid_preset", PANEL_SOLID_FIELDS,
                          "panelsolid", apply_panel_solid_preset)

            inputs = PanelSolidInputs(P_N=P_N, L_mm=back_mm, w_mm=w_mm, E_eff_MPa=E_eff_MPa,
                                       t_total_mm=t_total_mm, sigma_allow_MPa=sigma_allow_MPa)
            outputs = panel_solid_calc(inputs)
            passed = outputs["MS"] >= 0
            thickness = t_total_mm

    with left:
        st.subheader("Results")
        if is_sandwich:
            margin_metrics(outputs, ["MS_face", "MS_shear", "MS_wrinkle"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Total deflection (mm)", f"{outputs['delta_total_mm']:.2f}")
            c2.metric("Facesheet stress (MPa)", f"{outputs['sigma_face_MPa']:.1f}")
            c3.metric("Core shear stress (MPa)", f"{outputs['tau_core_MPa']:.2f}")
        else:
            margin_metrics(outputs, ["MS"])
            c1, c2 = st.columns(2)
            c1.metric("Deflection (mm)", f"{outputs['delta_mm']:.2f}")
            c2.metric("Bending stress (MPa)", f"{outputs['sigma_MPa']:.1f}")

        st.subheader("3D model — seat bottom panel")
        st.plotly_chart(panel_figure(front_mm, back_mm, length_mm, thickness, passed),
                         use_container_width=True)
        st.caption(f"{front_mm:.2f} mm front, {back_mm:.2f} mm back, {length_mm:.2f} mm long. "
                   f"Thickness shown to scale; green = all margins pass, red = at least one fails.")

    calc_fn = panel_calc if is_sandwich else panel_solid_calc
    dataclass_type = PanelInputs if is_sandwich else PanelSolidInputs
    sweep_section(list(asdict(inputs).keys()), asdict(inputs), calc_fn, dataclass_type, "panel")


# ---------------------------------------------------------------------------
# Tabs tab
# ---------------------------------------------------------------------------

def apply_tab_preset():
    preset = st.session_state.tab_presets[st.session_state.tab_preset]
    for f in TAB_FIELDS:
        st.session_state[f"tab_{f}"] = preset[f]
        st.session_state[f"tab_edit_{f}"] = preset[f]


def render_tabs_tab():
    left, right = st.columns([2, 1])

    with right:
        st.subheader("Material preset")
        st.selectbox("Tab material", list(st.session_state.tab_presets),
                     key="tab_preset", on_change=apply_tab_preset)
        preset = st.session_state.tab_presets[st.session_state.tab_preset]

        st.subheader("Variable inputs")
        F_total_N = st.number_input("Total design load (N)", value=3560.0, key="tab_F_total_N")
        N_tabs = st.number_input("Number of tabs", value=6, min_value=1, step=1, key="tab_N_tabs")
        b_mm = st.number_input("Tab width b (mm)", value=12.7, key="tab_b_mm")
        L_mm = st.number_input("Tab length L (mm)", value=19.05, key="tab_L_mm",
                                help="Cantilever moment arm: weld root to load point.")
        t_mm = st.number_input("Tab thickness t (mm)", value=6.35, key="tab_t_mm",
                                help="The actual/chosen thickness to test -- e.g. 6.35 mm "
                                     "(0.250in) stock. Change this to see whether a given "
                                     "shape passes or fails, same as the panel's inputs.")
        E_MPa = st.number_input("Tab modulus E (MPa)", value=preset["E_MPa"], key="tab_E_MPa")
        sigma_yield_MPa = st.number_input("Tab yield stress (MPa)",
                                           value=preset["sigma_yield_MPa"], key="tab_sigma_yield_MPa")
        FS_target = st.number_input("Target factor of safety", value=2.5, key="tab_FS_target")
        leg_size_mm = st.number_input("Weld leg size (mm)", value=3.0, key="tab_leg_size_mm")
        weld_length_mm = st.number_input("Weld length (mm)", value=25.4, key="tab_weld_length_mm")
        tau_weld_allow_MPa = st.number_input("Weld allowable shear (MPa)",
                                              value=preset["tau_weld_allow_MPa"],
                                              key="tab_tau_weld_allow_MPa")

        st.subheader("Fillet / hole (optional)")
        Kt_fillet = st.number_input("Root stress concentration Kt", value=1.0, min_value=1.0,
                                     key="tab_Kt_fillet",
                                     help="1.0 = no concentration modeled (a generous fillet, "
                                          "or plain beam theory). Increase toward a sharp "
                                          "corner -- see Peterson's Stress Concentration "
                                          "Factors for the real value at your r/t ratio.")
        fillet_radius_mm = st.number_input("Fillet radius, for the 3D view only (mm)", value=0.0,
                                            min_value=0.0, key="tab_fillet_radius_mm",
                                            help="Visualization only -- there's no verified "
                                                 "formula here linking radius to Kt, so this "
                                                 "doesn't feed the math above. Set Kt directly "
                                                 "once you know the real value for your radius.")
        hole_diameter_mm = st.number_input("Hole diameter (mm)", value=0.0, min_value=0.0,
                                            key="tab_hole_diameter_mm",
                                            help="0 = no hole modeled. Set this if the panel "
                                                 "bolts to the tab through it.")
        hole_position_from_root_mm = st.number_input("Hole position from root (mm)", value=0.0,
                                                       min_value=0.0, key="tab_hole_position_mm",
                                                       help="Distance from the weld root to "
                                                            "the hole center.")
        Kt_hole = st.number_input("Hole stress concentration Kt", value=2.0, min_value=1.0,
                                   key="tab_Kt_hole",
                                   help="PLACEHOLDER -- typical value for a round hole in "
                                        "bending. Verify against Peterson's charts for your "
                                        "actual d/b ratio. Only applied when a hole diameter "
                                        "is set above.")

        preset_editor("tab_presets", "tab_preset", TAB_FIELDS, "tab", apply_tab_preset)

    inputs = TabInputs(F_total_N=F_total_N, N_tabs=N_tabs, b_mm=b_mm, L_mm=L_mm, t_mm=t_mm,
                        E_MPa=E_MPa, sigma_yield_MPa=sigma_yield_MPa, FS_target=FS_target,
                        leg_size_mm=leg_size_mm, weld_length_mm=weld_length_mm,
                        tau_weld_allow_MPa=tau_weld_allow_MPa, Kt_fillet=Kt_fillet,
                        hole_diameter_mm=hole_diameter_mm,
                        hole_position_from_root_mm=hole_position_from_root_mm, Kt_hole=Kt_hole)
    outputs = tabs_calc(inputs)
    passed = all(outputs[k] >= 0 for k in ("MS_bend", "MS_hole", "MS_weld"))

    with left:
        st.subheader("Results")
        margin_metrics(outputs, ["MS_bend", "MS_hole", "MS_weld"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Root bending stress (MPa)", f"{outputs['sigma_bend_effective_MPa']:.1f}")
        c2.metric("Hole stress (MPa)", f"{outputs['sigma_hole_MPa']:.1f}")
        c3.metric("Weld shear stress (MPa)", f"{outputs['tau_weld_MPa']:.2f}")
        c4, c5 = st.columns(2)
        c4.metric("Tab-tip deflection (mm)", f"{outputs['delta_mm']:.4f}")
        c5.metric("Nominal bending stress, no Kt (MPa)", f"{outputs['sigma_bend_MPa']:.1f}")
        st.caption(f"Minimum thickness that would exactly hit the target FS: "
                   f"{outputs['t_min_required_mm']:.3f} mm (reference only, ignores Kt/hole — "
                   f"the checks above use the {t_mm:.3f} mm you entered).")
        if hole_diameter_mm <= 0:
            st.caption("MS_hole currently equals the root check (no hole set), so it's "
                       "redundant with MS_bend until you set a hole diameter above.")

        st.subheader("3D model — tab")
        st.plotly_chart(tab_figure(b_mm, t_mm, L_mm, passed,
                                    hole_diameter_mm=hole_diameter_mm,
                                    hole_position_mm=hole_position_from_root_mm,
                                    fillet_radius_mm=fillet_radius_mm),
                         use_container_width=True)
        st.caption("Thickness shown is the value you entered, not a solved value. Green = "
                   "bending, hole, and weld margins all pass, red = any fails. The hole is "
                   "shown as an open cutout (dark) at the position/diameter you set; the "
                   "fillet rounds the root's top and bottom edges (visualization only -- "
                   "set Kt above for its effect on the math).")

    sweep_section(list(asdict(inputs).keys()), asdict(inputs), tabs_calc, TabInputs, "tabs")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Seat Bottom Calculator", layout="wide")
st.title("Seat Bottom Calculator")

tab_panel, tab_tabs = st.tabs(["Seat Bottom Panel", "Tabs"])
with tab_panel:
    render_panel_tab()
with tab_tabs:
    render_tabs_tab()
