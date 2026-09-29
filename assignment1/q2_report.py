"""Equation-led Question 2 layout, preserving the recorded experiment results."""

import csv
import math

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, Table, TableStyle

from math_layout import MathBlock, equation, matrix


def _scientific(value):
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _table(headers, rows, widths):
    table = Table([headers, *rows], colWidths=[w * inch for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "BodyBold"),
        ("FONTNAME", (0, 1), (-1, -1), "Body"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    table.spaceAfter = 7
    return table


def build_q2_story(sim_dir, styles, swing_plot, double_plot, model_plot):
    def body(text):
        return Paragraph(text, styles["BodyCustom"])

    def heading(text):
        return Paragraph(text, styles["H1Custom"])

    def subheading(text):
        return Paragraph(text, styles["H2Custom"])

    def caption(text):
        return Paragraph(text, styles["Small"])

    with (sim_dir / "results" / "experiment_results.csv").open(newline="") as stream:
        single = list(csv.DictReader(stream))
    balance = [row for row in single
               if row["controller"] == "lqr" and float(row["offset_rad"]) != math.pi]
    swing = next(row for row in single if row["controller"] == "hybrid")
    with (sim_dir / "results" / "double_cartpole_results.csv").open(newline="") as stream:
        double = list(csv.DictReader(stream))
    with (sim_dir / "results" / "world_model_bonus_summary.csv").open(newline="") as stream:
        bonus = dict(csv.reader(stream))

    story = [
        heading("Question 2 - Model and Control the Cart-Pole"),
        subheading("A. Linearization about the upright equilibrium"),
        body("Use the simulator's state order and define deviations from the upright equilibrium:"),
        equation(r"\mathbf{s}=[x,\dot{x},\dot{\theta},\theta]^{\mathsf{T}},\qquad"
                 r"\mathbf{s}_{\star}=[0,0,0,\pi]^{\mathsf{T}}", number=1),
        equation(r"\phi=\theta-\pi,\qquad"
                 r"\boldsymbol{\delta}=[x,\dot{x},\dot{\theta},\phi]^{\mathsf{T}}"),
        body("The supplied nonlinear dynamics can be written with a shared denominator:"),
        equation(r"D(\theta)=4(M+m)-3m\cos^2\theta"),
        equation(r"\ddot{x}=\frac{2ml\dot{\theta}^{2}\sin\theta"
                 r"+3mg\sin\theta\cos\theta+4(u-b\dot{x})}{D(\theta)}", number=2),
        equation(r"\ddot{\theta}=-\frac{3\left[ml\dot{\theta}^{2}\sin\theta\cos\theta"
                 r"+2(M+m)g\sin\theta+2(u-b\dot{x})\cos\theta\right]}{lD(\theta)}"),
        body("Near upright, keep only first-order terms. The velocity-squared products are higher order:"),
        equation(r"\sin(\pi+\phi)\simeq-\phi,\quad"
                 r"\cos(\pi+\phi)\simeq-1,\quad D_0=4(M+m)-3m"),
        equation(r"\ddot{x}\simeq\frac{3mg\phi-4b\dot{x}+4u}{D_0},\qquad"
                 r"\ddot{\theta}\simeq\frac{6(M+m)g\phi-6b\dot{x}+6u}{lD_0}", number=3),
        equation(r"\dot{\boldsymbol{\delta}}=A\boldsymbol{\delta}+Bu"),
        equation(r"A=", matrix([
            [0, 1, 0, 0],
            [0, r"-\frac{4b}{D_0}", 0, r"\frac{3mg}{D_0}"],
            [0, r"-\frac{6b}{lD_0}", 0, r"\frac{6(M+m)g}{lD_0}"],
            [0, 0, 1, 0],
        ]), r"B=", matrix([[0], [r"\frac{4}{D_0}"], [r"\frac{6}{lD_0}"], [0]]),
                 size=11, number=4),
        body("With M = m = l = 0.5 and g = 9.82, the denominator is 2.5. The handout uses "
             "b = 0.1, giving damping entries -0.16 and -0.48. The executable simulator uses "
             "b = 1.0; I matched that value for the controller and experiments:"),
        equation(r"A=", matrix([[0, 1, 0, 0], [0, -1.6, 0, 5.892],
                                [0, -4.8, 0, 47.136], [0, 0, 1, 0]]),
                 r"B=", matrix([[0], [1.6], [4.8], [0]]), size=11, number=5),
        PageBreak(),
        heading("B. LQR design and balancing"),
        body("I gave angle error the largest weight. Cart-position and velocity penalties encourage "
             "recentering and damping, while the input penalty discourages excessive force:"),
        equation(r"Q=\mathrm{diag}(2,1,2,120),\qquad R=[0.2]", number=6),
        body("Solve the continuous algebraic Riccati equation and use its stabilizing solution:"),
        equation(r"A^{\mathsf{T}}P+PA-PBR^{-1}B^{\mathsf{T}}P+Q=0,\quad"
                 r"K=R^{-1}B^{\mathsf{T}}P", number=7),
        equation(r"K=[-3.1623,\,-6.4776,\,7.2892,\,47.0906]"),
        equation(r"u_{\mathrm{LQR}}=-K\boldsymbol{\delta}=K(\mathbf{s}_{\star}-\mathbf{s}),"
                 r"\qquad u=\mathrm{clip}(u_{\mathrm{LQR}},-30,30)", number=8),
        body("The angular error is wrapped to the principal interval. All closed-loop eigenvalues "
             "have negative real parts, establishing local asymptotic stability of the linear model:"),
        equation(r"\lambda(A-BK)=\{-16.7176,\,-7.7261,\,-0.8903\pm0.6012\,i\}"),
        subheading("B(i). Default start and initial-offset tests"),
        body("Each deterministic trial uses the supplied nonlinear simulator for 20 s, with "
             "a 0.005 s time step. Q, R, K, and the 30 N force limit are fixed across trials. "
             "Success requires both conditions throughout the final five seconds:"),
        equation(r"|\phi(t)|<0.10\ \mathrm{rad},\qquad|x(t)|<0.50\ \mathrm{m},"
                 r"\qquad 15\leq t\leq20\ \mathrm{s}", number=9),
        body("Capture time is the first sampled time satisfying both angular conditions:"),
        equation(r"|\phi|<0.10\ \mathrm{rad},\qquad"
                 r"|\dot{\theta}|<0.25\ \mathrm{rad}\,\mathrm{s}^{-1}", size=10.5),
    ]
    labels = ["default (π/40)", "0.01", "0.10", "π/8", "π/4"]
    story += [
        _table(["initial offset", "capture (s)", "peak |u| (N)", "tail max |φ| (rad)", "result"],
               [[label, f"{float(row['capture_time_s']):.3f}",
                 f"{float(row['max_force_N']):.2f}",
                 f"{float(row['tail_max_angle_error_rad']):.2e}",
                 "PASS" if row["success"] == "True" else "FAIL"]
                for label, row in zip(labels, balance)], [1.12, 0.91, 1.0, 1.42, 0.62]),
        body("The default start and all four requested offsets balance. The π/4 trial saturates "
             "at 30 N and takes 1.205 s to meet the capture criterion; smaller tested errors remain "
             "unsaturated. The largest successful tested offset is π/4 (0.785 rad), with all other "
             "initial deviations zero. These sampled results do not certify a continuous range or "
             "global stability; the Riccati proof is local and saturation changes the nonlinear loop."),
        PageBreak(),
        heading("C. Swing-up from the downward configuration"),
        subheading("C(i). Initial control performance"),
        body("An initial offset of π starts the pole at an angle of 2π, physically downward. Pure LQR "
             "saturates and fails to settle upright during 20 s; its final-window maximum angular "
             "error is 3.1411 rad. Changing Q and R can adjust local aggressiveness, but LQR "
             "does not provide a strategy for building the energy needed to swing up."),
        subheading("C(ii). Changes explored"),
        body("I used energy shaping away from upright and LQR inside a capture region. For a "
             "uniform rod, the pole's energy relative to downward and its upright target are:"),
        equation(r"E=\frac{ml^2}{6}\dot{\theta}^{2}+\frac{mgl}{2}(1-\cos\theta),"
                 r"\qquad E_{\star}=mgl", number=10),
        equation(r"u_{\mathrm{swing}}=-k_E(E_{\star}-E)\,\sigma(\dot{\theta}\cos\theta)"
                 r"-k_xx-k_v\dot{x}", number=11),
        equation(r"k_E=40,\qquad k_x=1,\qquad k_v=2"),
        body("The phase function σ is the sign of its argument; for magnitude below 10<super>-8</super>, "
             "it is set to +1 to start motion from rest. Force remains clipped to ±30 N. The "
             "controller enters LQR when both capture conditions hold, and returns to energy "
             "shaping if the angle exceeds the release threshold:"),
        equation(r"\mathrm{capture}:\ |\phi|<0.42\ \mathrm{rad},\quad"
                 r"|\dot{\theta}|<3.5\ \mathrm{rad}\,\mathrm{s}^{-1};"
                 r"\qquad\mathrm{release}:\ |\phi|>0.65\ \mathrm{rad}", size=10.5, number=12),
        subheading("C(iii). Best swing-up performance"),
        body("The hybrid controller first meets the angular capture criterion at 1.340 s and passes "
             "the 20 s success test. Its final-five-second errors and peak force are:"),
        equation(r"\max|\phi|=" + _scientific(swing['tail_max_angle_error_rad']),
                 r"\mathrm{rad},\qquad\max|x|=" + _scientific(swing['tail_max_cart_position_m']),
                 r"\mathrm{m},\qquad\max|u|=30\ \mathrm{N}", size=10.5),
        Image(str(swing_plot), width=6.25 * inch, height=2.97 * inch),
        caption("Figure 1. Downward-start comparison. An upright score of +1 means upright and -1 "
                "means downward. Energy shaping swings up the pole, then LQR recenters the cart."),
        body("This is my best single-pole swing-up result in deterministic simulation. "
             "Actuator delay, sensor noise, and model mismatch were not tested."),
        PageBreak(),
        heading("D. DoubleCartpole bonus"),
        body("Use both pole angles measured from downward. The six-state upright deviation is:"),
        equation(r"\phi_i=\theta_i-\pi,\qquad\boldsymbol{\delta}_d="
                 r"[x,\dot{x},\dot{\theta}_1,\dot{\theta}_2,\phi_1,\phi_2]^{\mathsf{T}}", number=13),
        body("At upright, the simulator's coupled acceleration equations reduce to the following "
             "mass-matrix system. Together with the kinematic equations, this gives the linear model:"),
        equation(r"H_0\mathbf{a}=\mathbf{r},\qquad"
                 r"\mathbf{a}=[\ddot{x},\ddot{\theta}_1,\ddot{\theta}_2]^{\mathsf{T}}"),
        equation(r"H_0=", matrix([[3, 0.9, 0.3], [4.5, 2.4, 0.9], [3, 1.8, 1.2]]),
                 r"\mathbf{r}=", matrix([[r"-0.2\dot{x}+2u"], [r"44.19\phi_1"], [r"29.46\phi_2"]]),
                 number=14),
        equation(r"\mathbf{a}=H_0^{-1}\mathbf{r},\qquad"
                 r"\dot{\boldsymbol{\delta}}_d=A_d\boldsymbol{\delta}_d+B_du"),
        equation(r"Q_d=\mathrm{diag}(2,1,2,2,120,120),\qquad R_d=[0.2]"),
        equation(r"K_d=[3.1623,\,6.8993,\,0.8285,\,27.1835,\,-135.5356,\,189.0925]", number=15),
        equation(r"u=\mathrm{clip}(-K_d\boldsymbol{\delta}_d,-40,40),\qquad"
                 r"\mathrm{Re}\{\lambda(A_d-B_dK_d)\}<0"),
        body("A central-difference check agrees with the derived Jacobians to about 10<super>-8</super>. "
             "Trials use the same duration, time step, and final-window success test as part B. "
             "The angular success and capture conditions must hold for both poles."),
        _table(["start offset", "capture (s)", "tail max error (rad)", "peak |x| (m)", "result"],
               [[label, f"{float(row['capture_time_s']):.2f}" if row['capture_time_s'] else "--",
                 f"{max(float(row['tail_max_pole1_error_rad']), float(row['tail_max_pole2_error_rad'])):.3g}",
                 f"{float(row['max_cart_position_m']):.3g}",
                 "PASS" if row['success'] == "True" else "FAIL"]
                for label, row in zip(["default (π/40)", "0.01", "0.10", "π/8", "0.42", "π/4", "π"], double)],
               [1.12, 0.85, 1.5, 1.02, 0.62]),
        body("Both poles balance from the default start through the tested π/8 offset with a 40 N cap. "
             "The π/8 trial briefly moves the cart 1.23 m before recentering. A 30 N cap failed at π/8. "
             "The largest successful tested offset is 0.393 rad; the next tested offset, 0.42 rad, "
             "fails, as do π/4 and the downward start. No untested initial conditions are claimed."),
        Image(str(double_plot), width=6.25 * inch, height=2.42 * inch),
        caption("Figure 2. Both pole-angle errors and cart position during the π/8 recovery."),
        PageBreak(),
        heading("E. Research bonus - learned world model"),
        body("I collected 7,680 training and 1,920 held-out transitions using randomized initial "
             "states and forces. Each row stores the next state, current state, and force. "
             "Least squares fits coefficients of trigonometric features from the transitions; "
             "no simulator constants are used in the fit:"),
        equation(r"\widehat{\ddot{x}}=\frac{c_1\dot{\theta}^{2}\sin\theta"
                 r"+c_2\sin\theta\cos\theta+c_3u+c_4\dot{x}}{1-c_5\cos^2\theta}", number=16),
        equation(r"\widehat{\ddot{\theta}}=\frac{p_1\dot{\theta}^{2}\sin\theta\cos\theta"
                 r"+p_2\sin\theta+p_3u\cos\theta+p_4\dot{x}\cos\theta}{1-p_5\cos^2\theta}"),
        body("Fourth-order integration predicts the next state. A 50 ms receding-horizon controller "
             "compares candidate forces held constant over ten learned-model steps, adds the LQR "
             "terminal value, and replans after each simulator step:"),
        equation(r"u^{\star}=\underset{u\in\mathcal{U}}{\mathrm{arg\,min}}"
                 r"\left[\sum_{k=1}^{10}\Delta t\left(\boldsymbol{\delta}_k^{\mathsf{T}}Q"
                 r"\boldsymbol{\delta}_k+0.2u^2\right)"
                 r"+\boldsymbol{\delta}_{10}^{\mathsf{T}}P\boldsymbol{\delta}_{10}\right]", number=17),
        equation(r"\Delta t=0.005\ \mathrm{s},\qquad"
                 r"\mathcal{U}=\{-30,-28,\ldots,30\}\cup\{u_{\mathrm{LQR}}\}"),
        body("On 12 held-out 0.8 s rollouts driven by recorded forces, the prediction errors are:"),
        _table(["state variable", "one-step RMSE", "0.8 s open-loop RMSE"],
               [["cart position (m)", f"{float(bonus['heldout_one_step_rmse_x']):.2e}",
                 f"{float(bonus['heldout_open_loop_0p8s_rmse_x']):.2e}"],
                ["pole angle (rad)", f"{float(bonus['heldout_one_step_rmse_theta']):.2e}",
                 f"{float(bonus['heldout_open_loop_0p8s_rmse_theta']):.2e}"]], [1.6, 1.55, 1.75]),
        body("The lookahead controller balances all three tested offsets in 8 s trials. "
             "The table reports maximum errors during the final two seconds:"),
        _table(["start offset", "tail max |φ| (rad)", "tail max |x| (m)", "peak |u| (N)"],
               [[label,
                 f"{float(bonus[f'lookahead_offset_{offset:.6f}_tail_max_angle_error_rad']):.3g}",
                 f"{float(bonus[f'lookahead_offset_{offset:.6f}_tail_max_cart_position_m']):.3g}",
                 f"{float(bonus[f'lookahead_offset_{offset:.6f}_peak_force_N']):.1f}"]
                for label, offset in [("0.1", 0.1), ("π/8", math.pi / 8), ("π/4", math.pi / 4)]],
               [1.0, 1.5, 1.3, 1.1]),
        Image(str(model_plot), width=6.05 * inch, height=2.30 * inch),
        caption("Figure 3. One unseen trajectory: predicted pole rate follows the simulator; "
                "the lower panel shows accumulated angle error in milliradians."),
        body("This demonstrates learned-model use in control; it is not shown to outperform LQR "
             "or to swing up from downward."),
        caption("Reproduce the experiments in assignment1/TeachingCartpole with "
                "python3 experiment.py, python3 double_experiment.py, and python3 world_model_bonus.py. "
                "Simulator: D. Meger et al., https://github.com/dmeger/TeachingCartpole."),
    ]
    # Keep the double-pole derivation, table, and trajectory on one page.
    # Equation glyphs retain their size; only the surrounding white space changes.
    double_section = False
    for flowable in story:
        if isinstance(flowable, Paragraph) and flowable.getPlainText().startswith("D. DoubleCartpole"):
            double_section = True
        elif isinstance(flowable, PageBreak):
            double_section = False
        if double_section and isinstance(flowable, MathBlock):
            flowable.spaceBefore = 1
            flowable.spaceAfter = 3
    return story
