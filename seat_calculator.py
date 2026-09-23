"""
seat_calculator.py -- Seat bottom panel and tab structural calculator.

Mirrors the math in the team's Notion "Seat Bottom & Back Research" doc:
sandwich-beam panel screening (skin/core split, or a simplified solid/
combined-property mode) plus welded flat-cantilever tab sizing.

Single-point calculation:
    python seat_calculator.py panel [--P_N 3560] [--L_mm 432.97] ...
    python seat_calculator.py panel-solid [--P_N 3560] [--E_eff_MPa 20000] ...
    python seat_calculator.py tabs [--F_total_N 3560] [--N_tabs 6] ...

Run with -h on any subcommand to see every adjustable variable and its
current default (the defaults match the Notion calculator's baseline row).

Parameter sweep (for testing tab shapes / panel constructions):
    python seat_calculator.py sweep --calc tabs --var b_mm --start 8 --stop 20 --steps 25
    python seat_calculator.py sweep --calc panel --var t_c_mm --start 3 --stop 15 --steps 25

    Any other variable can still be overridden for the sweep, e.g.:
    python seat_calculator.py sweep --calc tabs --var N_tabs --start 2 --stop 8 --steps 7 --L_mm 25

Sweeps write a CSV of every input/output at each step, and a PNG plot
(one panel per output) if matplotlib is installed.

Notes carried over from the Notion doc:
  - Panel `w_mm` is the loaded contact width (e.g. a shoe patch), not the
    physical panel width -- see the beam-strip caveat below.
  - Panel `L_mm` defaults to the back span (432.97 mm), which bounds the
    front span for every L-dependent check (core shear doesn't depend on L
    at all, so span choice only matters for deflection/face stress).
  - Tab formulas assume a flat, prismatic bar -- no gusset, no taper, no
    holes. If a gusset is added later, this stays a conservative check
    (max moment is always at the fixed root, and a gusset only stiffens
    the root), it just stops being the *precise* answer.
  - The sandwich beam-strip formulas bound the true 2D plate response,
    they don't equal it -- treat panel results as a screening bound
    unless cross-checked in Ansys or a Roark's plate solution.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass, fields, asdict


# ---------------------------------------------------------------------------
# Panel -- sandwich beam (skin/core split)
# ---------------------------------------------------------------------------

@dataclass
class PanelInputs:
    P_N: float = 3560.0        # design load (N) -- 800 lbf dynamic egress load
    L_mm: float = 432.97       # span (mm) -- back span governs (see module docstring)
    w_mm: float = 100.0        # loaded width (mm) -- foot/shoe contact patch, NOT panel width
    t_f_mm: float = 0.55       # facesheet (skin) thickness (mm)
    E_f_MPa: float = 55000.0   # facesheet modulus (MPa)
    t_c_mm: float = 6.35       # core thickness (mm)
    G_c_MPa: float = 22.0      # core shear modulus (MPa)
    E_c_MPa: float = 3000.0    # core modulus (MPa) -- wrinkling check only
    sigma_allow_MPa: float = 500.0  # facesheet allowable stress (MPa)
    tau_allow_MPa: float = 0.9      # core allowable shear stress (MPa)


def panel_calc(p: PanelInputs) -> dict:
    D = p.E_f_MPa * p.t_f_mm * (p.t_c_mm + p.t_f_mm) ** 2 / 2
    S = p.G_c_MPa * (p.t_c_mm + p.t_f_mm) ** 2 / p.t_c_mm
    delta_bending = p.P_N * p.L_mm ** 3 / (48 * D * p.w_mm)
    delta_shear = p.P_N * p.L_mm / (4 * S * p.w_mm)
    delta_total = delta_bending + delta_shear
    M = p.P_N * p.L_mm / 4
    sigma_face = M / (p.w_mm * p.t_f_mm * (p.t_c_mm + p.t_f_mm))
    V = p.P_N / 2
    tau_core = V / (p.w_mm * (p.t_c_mm + p.t_f_mm))
    sigma_wrinkle = 0.5 * (p.E_f_MPa * p.E_c_MPa * p.G_c_MPa) ** (1 / 3)
    return {
        "D_bending_stiffness": D,
        "S_shear_stiffness": S,
        "delta_bending_mm": delta_bending,
        "delta_shear_mm": delta_shear,
        "delta_total_mm": delta_total,
        "sigma_face_MPa": sigma_face,
        "tau_core_MPa": tau_core,
        "sigma_wrinkle_MPa": sigma_wrinkle,
        "MS_face": p.sigma_allow_MPa / sigma_face - 1,
        "MS_shear": p.tau_allow_MPa / tau_core - 1,
        "MS_wrinkle": sigma_wrinkle / sigma_face - 1,
    }


# ---------------------------------------------------------------------------
# Panel -- solid/combined mode (one lumped modulus + total thickness,
# for when a supplier only gives an effective panel property, not a
# separate skin/core split)
# ---------------------------------------------------------------------------

@dataclass
class PanelSolidInputs:
    P_N: float = 3560.0
    L_mm: float = 432.97
    w_mm: float = 100.0
    E_eff_MPa: float = 20000.0
    t_total_mm: float = 7.5
    sigma_allow_MPa: float = 300.0


def panel_solid_calc(p: PanelSolidInputs) -> dict:
    I = p.w_mm * p.t_total_mm ** 3 / 12
    delta = p.P_N * p.L_mm ** 3 / (48 * p.E_eff_MPa * I)
    M = p.P_N * p.L_mm / 4
    c = p.t_total_mm / 2
    sigma = M * c / I
    return {
        "I_mm4": I,
        "delta_mm": delta,
        "sigma_MPa": sigma,
        "MS": p.sigma_allow_MPa / sigma - 1,
    }


# ---------------------------------------------------------------------------
# Tabs -- welded flat cantilever
# ---------------------------------------------------------------------------

@dataclass
class TabInputs:
    F_total_N: float = 3560.0   # total design load (N), shared with the panel's P_N
    N_tabs: float = 6           # number of tabs sharing the load
    b_mm: float = 12.7          # tab width (mm)
    L_mm: float = 19.05         # tab length / moment arm (mm), root to load point
    t_mm: float = 6.35          # ACTUAL/chosen tab thickness (mm) -- e.g. 0.250in stock.
                                 # This is a real input now, not solved-for -- change it
                                 # to test whether a given shape passes or fails.
    E_MPa: float = 205000.0     # tab material modulus (MPa)
    sigma_yield_MPa: float = 460.0  # tab material yield stress (MPa)
    FS_target: float = 2.5      # target factor of safety
    leg_size_mm: float = 3.0    # weld leg size (mm)
    weld_length_mm: float = 25.4  # weld length (mm), both sides of base per rule
    tau_weld_allow_MPa: float = 150.0  # weld metal allowable shear (MPa) -- PLACEHOLDER,
                                        # replace with your filler metal's rated allowable
    Kt_fillet: float = 1.0      # stress concentration factor at the root. 1.0 = idealized/
                                 # no concentration modeled (a generous fillet radius, or the
                                 # plain beam theory this whole tool otherwise assumes).
                                 # Increase toward a sharp corner -- see Peterson's Stress
                                 # Concentration Factors for the real value at your r/t ratio.
    hole_diameter_mm: float = 0.0  # 0 = no hole modeled. If the panel bolts to the tab
                                    # through it, set this to the hole's diameter.
    hole_position_from_root_mm: float = 0.0  # distance from the weld root to the hole
    Kt_hole: float = 2.0        # stress concentration around the hole edge -- PLACEHOLDER,
                                 # a typical value for a round hole in bending; verify against
                                 # Peterson's charts for your actual d/b ratio.


def tabs_calc(t: TabInputs) -> dict:
    """Checks a given tab shape (t.t_mm is a real input) against the load,
    the same way panel_calc checks a given panel construction -- this can
    genuinely fail, unlike solving t backward from the allowable stress."""
    F_tab = t.F_total_N / t.N_tabs
    sigma_allow = t.sigma_yield_MPa / t.FS_target
    I = t.b_mm * t.t_mm ** 3 / 12
    delta = F_tab * t.L_mm ** 3 / (3 * t.E_MPa * I)

    # Root bending check, with an optional fillet stress concentration factor.
    sigma_bend = 6 * F_tab * t.L_mm / (t.b_mm * t.t_mm ** 2)
    sigma_bend_effective = t.Kt_fillet * sigma_bend
    MS_bend = sigma_allow / sigma_bend_effective - 1

    # Hole check, using a net-section width and a stress concentration factor
    # around the hole edge. With hole_diameter_mm == 0 this reduces to the
    # same nominal bending check as the root (Kt forced to 1, full width used),
    # so it's harmless to leave enabled even when there's no hole.
    hole_active = t.hole_diameter_mm > 0
    b_net = max(t.b_mm - t.hole_diameter_mm, 0.1)
    M_hole = F_tab * max(t.L_mm - t.hole_position_from_root_mm, 0.0)
    kt_hole_used = t.Kt_hole if hole_active else 1.0
    sigma_hole = kt_hole_used * 6 * M_hole / (b_net * t.t_mm ** 2)
    MS_hole = sigma_allow / sigma_hole - 1 if sigma_hole > 0 else float("inf")

    tau_weld = F_tab / (0.707 * t.leg_size_mm * t.weld_length_mm)
    # informational only -- the minimum thickness that would exactly hit
    # sigma_allow, for reference when picking a real stock thickness
    t_min_required_mm = math.sqrt(6 * F_tab * t.L_mm / (t.b_mm * sigma_allow))
    return {
        "F_tab_N": F_tab,
        "sigma_allow_MPa": sigma_allow,
        "I_mm4": I,
        "delta_mm": delta,
        "sigma_bend_MPa": sigma_bend,
        "sigma_bend_effective_MPa": sigma_bend_effective,
        "MS_bend": MS_bend,
        "sigma_hole_MPa": sigma_hole,
        "MS_hole": MS_hole,
        "tau_weld_MPa": tau_weld,
        "MS_weld": t.tau_weld_allow_MPa / tau_weld - 1,
        "t_min_required_mm": t_min_required_mm,
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

CALCS = {
    "panel": (PanelInputs, panel_calc),
    "panel-solid": (PanelSolidInputs, panel_solid_calc),
    "tabs": (TabInputs, tabs_calc),
}


def add_dataclass_args(subparser, dataclass_type):
    for f in fields(dataclass_type):
        subparser.add_argument(f"--{f.name}", type=type(f.default), default=None,
                                help=f"default: {f.default}")


def build_inputs(dataclass_type, args):
    kwargs = {}
    for f in fields(dataclass_type):
        val = getattr(args, f.name, None)
        if val is not None:
            kwargs[f.name] = val
    return dataclass_type(**kwargs)


def print_result(name, inputs, outputs):
    print(f"\n=== {name} ===")
    print("-- inputs --")
    for k, v in asdict(inputs).items():
        print(f"  {k:20s} = {v}")
    print("-- outputs --")
    for k, v in outputs.items():
        flag = ""
        if k.startswith("MS") or k == "MS":
            flag = "  <-- FAILS (negative margin)" if v < 0 else "  ok"
        print(f"  {k:20s} = {v:.4g}{flag}")


def run_single(calc_name, args):
    dataclass_type, calc_fn = CALCS[calc_name]
    inputs = build_inputs(dataclass_type, args)
    outputs = calc_fn(inputs)
    print_result(calc_name, inputs, outputs)


def run_sweep(args):
    dataclass_type, calc_fn = CALCS[args.calc]
    field_names = {f.name for f in fields(dataclass_type)}
    if args.var not in field_names:
        sys.exit(f"--var must be one of: {sorted(field_names)}")

    base_kwargs = {}
    for f in fields(dataclass_type):
        val = getattr(args, f.name, None)
        if val is not None:
            base_kwargs[f.name] = val

    if args.steps < 2:
        sys.exit("--steps must be >= 2")
    step_size = (args.stop - args.start) / (args.steps - 1)
    values = [args.start + i * step_size for i in range(args.steps)]

    rows = []
    for v in values:
        kwargs = dict(base_kwargs)
        kwargs[args.var] = v
        inputs = dataclass_type(**kwargs)
        outputs = calc_fn(inputs)
        row = {**asdict(inputs), **outputs}
        rows.append(row)

    out_csv = args.out or f"sweep_{args.calc}_{args.var}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

    output_keys = list(calc_fn(dataclass_type(**base_kwargs) if base_kwargs else dataclass_type()).keys())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(output_keys)
        ncols = 3
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() if n > 1 else [axes]
        x = [row[args.var] for row in rows]
        for ax, key in zip(axes, output_keys):
            y = [row[key] for row in rows]
            ax.plot(x, y, marker="o")
            ax.set_xlabel(args.var)
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.grid(True, alpha=0.3)
            if key.startswith("MS"):
                ax.axhline(0, color="red", linestyle="--", linewidth=1)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{args.calc}: sweep of {args.var}")
        fig.tight_layout()
        out_png = args.out_png or f"sweep_{args.calc}_{args.var}.png"
        fig.savefig(out_png, dpi=150)
        print(f"Wrote plot to {out_png}")
    except ImportError:
        print("matplotlib not installed -- skipping plot (CSV was still written). "
              "Install with: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=False)

    for name, (dataclass_type, _fn) in CALCS.items():
        p = sub.add_parser(name, help=f"single-point {name} calculation")
        add_dataclass_args(p, dataclass_type)

    sweep_p = sub.add_parser("sweep", help="sweep one variable and write CSV + plot")
    sweep_p.add_argument("--calc", choices=list(CALCS.keys()), required=True)
    sweep_p.add_argument("--var", required=True, help="which input variable to sweep")
    sweep_p.add_argument("--start", type=float, required=True)
    sweep_p.add_argument("--stop", type=float, required=True)
    sweep_p.add_argument("--steps", type=int, required=True)
    sweep_p.add_argument("--out", help="CSV output path")
    sweep_p.add_argument("--out-png", help="PNG output path")
    # allow overriding any field of any calc type for the sweep's held-fixed values
    all_fields = {}
    for _name, (dataclass_type, _fn) in CALCS.items():
        for f in fields(dataclass_type):
            all_fields[f.name] = f.default
    for fname, fdefault in all_fields.items():
        sweep_p.add_argument(f"--{fname}", type=type(fdefault), default=None,
                              help=f"override for sweep (default: {fdefault})")

    args = parser.parse_args()

    if args.command is None:
        # No subcommand given (e.g. run directly from an IDE with no args
        # configured) -- show all three baseline calculations instead of
        # just erroring, and point at how to do more.
        print("No subcommand given -- running all three baseline calculations.")
        print("Run with -h for full usage, e.g.:")
        print("  python seat_calculator.py tabs --N_tabs 4 --b_mm 15")
        print("  python seat_calculator.py sweep --calc tabs --var b_mm --start 8 --stop 20 --steps 13")
        for name, (dataclass_type, calc_fn) in CALCS.items():
            inputs = dataclass_type()
            outputs = calc_fn(inputs)
            print_result(name, inputs, outputs)
        return

    if args.command == "sweep":
        run_sweep(args)
    else:
        run_single(args.command, args)


if __name__ == "__main__":
    main()
