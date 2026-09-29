"""Build the final Assignment 1 PDF from verified experiment outputs."""

from pathlib import Path
import csv
import math
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SIM = ROOT / "TeachingCartpole"
sys.path.insert(0, str(SIM))

from cartpole_control import Controller, wrap_to_pi  # noqa: E402
from cartpole_envs import CartPole, DoubleCartPole  # noqa: E402


OUTPUT = ROOT / "output" / "pdf" / "COMP765_Assignment1.pdf"
PLOT = ROOT.parent / "tmp" / "pdfs" / "swingup_comparison.png"
MODEL_PLOT = ROOT.parent / "tmp" / "pdfs" / "world_model_validation.png"
DOUBLE_PLOT = ROOT.parent / "tmp" / "pdfs" / "double_cartpole_balance.png"
RESULTS = SIM / "results" / "experiment_results.csv"
BONUS = SIM / "results" / "world_model_bonus_summary.csv"
DOUBLE_RESULTS = SIM / "results" / "double_cartpole_results.csv"


def simulate(offset, hybrid, duration=8.0, dt=0.005):
    env = CartPole(initial_offset=offset)
    controller = Controller(hybrid=hybrid)
    rows = []
    for step in range(int(duration / dt)):
        state = env.get_state().copy()
        force = controller.compute_control(state)
        state = env.step(force, dt=dt).copy()
        rows.append(
            (step * dt, wrap_to_pi(state[3] - math.pi), state[0], force)
        )
    return np.asarray(rows)


def simulate_double(offset, duration=5.0, dt=0.005):
    start = [0.0, 0.0, 0.0, 0.0, math.pi, math.pi]
    env = DoubleCartPole(x_init=start, initial_offset=offset)
    controller = Controller(hybrid=False)
    rows = []
    for step in range(int(duration / dt)):
        state = env.get_state().copy()
        force = controller.compute_control(state)
        state = env.step(force, dt=dt).copy()
        rows.append((step * dt, state[0],
                     abs(wrap_to_pi(state[4] - math.pi)),
                     abs(wrap_to_pi(state[5] - math.pi)), force))
    return np.asarray(rows)


def build_plot():
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    pure = simulate(math.pi, False)
    hybrid = simulate(math.pi, True)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.4), sharex=True)
    axes[0].plot(pure[:, 0], np.cos(pure[:, 1]), color="#9a3412", lw=1.2, label="Pure LQR")
    axes[0].plot(hybrid[:, 0], np.cos(hybrid[:, 1]), color="#075985", lw=1.2, label="Hybrid")
    axes[0].axhline(1, color="#64748b", lw=0.7)
    axes[0].set_ylabel("upright score")
    axes[0].set_ylim(-1.1, 1.12)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[1].plot(pure[:, 0], pure[:, 2], color="#9a3412", lw=1.2)
    axes[1].plot(hybrid[:, 0], hybrid[:, 2], color="#075985", lw=1.2)
    axes[1].axhline(0, color="#64748b", lw=0.7)
    axes[1].set_ylabel("cart x (m)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, color="#dbe3ea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=190, bbox_inches="tight")
    plt.close(fig)

    trace = np.loadtxt(SIM / "results" / "world_model_heldout_rollout.csv",
                       delimiter=",", skiprows=1)
    t = np.arange(len(trace)) * 0.005
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 2.7), sharex=True)
    axes[0].plot(t, trace[:, 2], color="#0f3d73", lw=1.5, label="Simulator")
    axes[0].plot(t, trace[:, 6], color="#d97706", lw=1.2, ls="--", label="Learned model")
    axes[0].set_ylabel("rate (rad/s)")
    axes[0].legend(loc="lower left", frameon=False, ncol=2, fontsize=8)
    axes[1].plot(t, 1000 * (trace[:, 7] - trace[:, 3]), color="#0f766e", lw=1.2)
    axes[1].axhline(0, color="#64748b", lw=0.7)
    axes[1].set_ylabel("error (mrad)")
    axes[1].set_xlabel("held-out rollout time (s)")
    for ax in axes:
        ax.grid(True, color="#dbe3ea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_PLOT, dpi=190, bbox_inches="tight")
    plt.close(fig)

    double = simulate_double(math.pi / 8)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 2.75), sharex=True)
    axes[0].plot(double[:, 0], double[:, 2], color="#075985", lw=1.4,
                 label="Pole 1")
    axes[0].plot(double[:, 0], double[:, 3], color="#d97706", lw=1.2,
                 ls="--", label="Pole 2")
    axes[0].axhline(0.1, color="#64748b", lw=0.7, ls=":")
    axes[0].set_ylabel("angle error (rad)")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[1].plot(double[:, 0], double[:, 1], color="#0f766e", lw=1.3)
    axes[1].axhline(0, color="#64748b", lw=0.7)
    axes[1].set_ylabel("cart x (m)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, color="#dbe3ea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOUBLE_PLOT, dpi=190, bbox_inches="tight")
    plt.close(fig)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(0.7 * inch, 0.58 * inch, 7.8 * inch, 0.58 * inch)
    canvas.setFont("Body", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(0.7 * inch, 0.36 * inch, "COMP 765 - Assignment 1")
    canvas.drawRightString(7.8 * inch, 0.36 * inch, f"Page {doc.page}")
    canvas.restoreState()


def p(text, style):
    return Paragraph(text, style)


def matrix_table(rows, widths=None):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Courier"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe3ea")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_plot()

    font_regular = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    font_bold = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
    pdfmetrics.registerFont(TTFont("Body", font_regular))
    pdfmetrics.registerFont(TTFont("BodyBold", font_bold))

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            fontName="BodyBold",
            fontSize=21,
            leading=24,
            textColor=colors.HexColor("#173f73"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            fontName="Body",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1Custom",
            fontName="BodyBold",
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#173f73"),
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Custom",
            fontName="BodyBold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#075985"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            fontName="Body",
            fontSize=9.6,
            leading=12.5,
            textColor=colors.HexColor("#0f172a"),
            alignment=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            fontName="Body",
            fontSize=8.1,
            leading=10.2,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Equation",
            fontName="Body",
            fontSize=10.2,
            leading=13,
            leftIndent=16,
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=6,
        )
    )

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.66 * inch,
        bottomMargin=0.72 * inch,
        title="COMP 765 Assignment 1",
        author="COMP 765 student",
        subject="World models and cart-pole control",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=header_footer))

    B = styles["BodyCustom"]
    H1 = styles["H1Custom"]
    H2 = styles["H2Custom"]
    EQ = styles["Equation"]
    S = styles["Small"]
    story = []

    story += [
        Spacer(1, 0.08 * inch),
        p("Assignment 1: Intro to World Models and Control", styles["TitleCustom"]),
        p("COMP 765 - Fall 2026", styles["Subtitle"]),
        p("Question 1 - Profile of a World Model: DreamerV3", H1),
        p(
            "<b>What goes in and what comes out.</b> DreamerV3 is a model-based reinforcement-learning "
            "agent rather than a stand-alone video generator. At each environment step it consumes an "
            "observation (RGB pixels or a low-dimensional state vector), the previous action, reward, and "
            "episode-continuation flag. An encoder maps the observation to categorical stochastic variables "
            "<i>z</i><sub>t</sub>; a recurrent sequence model carries deterministic state <i>h</i><sub>t</sub>. "
            "Together they form the compact model state. From it, learned heads reconstruct the observation "
            "and predict reward and continuation, while the transition prior predicts the next latent state "
            "conditioned on actions. The actor outputs discrete or continuous actions and the critic outputs "
            "a distribution over future return [1].",
            B,
        ),
        p(
            "<b>Design principles.</b> The central idea is to learn behavior from imagined latent trajectories, "
            "not by rolling pixels forward for every policy update. Replay observations train a recurrent "
            "state-space model (RSSM); imagined rollouts of length 16 then train an actor-critic using predicted "
            "rewards, continuation probabilities, and bootstrapped lambda-returns. This separates representation "
            "learning from decision learning while retaining a differentiable, action-conditioned simulator. "
            "Unlike classical system identification, the latent state need not correspond to named physical "
            "variables. Unlike MuZero-style task-centric models, Dreamer reconstructs sensory inputs, so its "
            "representation is shaped by general observation structure as well as reward. Unlike online search "
            "or MPC, the deployed actor selects an action directly without look-ahead search [1].",
            B,
        ),
        p(
            "<b>What distinguishes V3.</b> DreamerV1 introduced analytic actor gradients through continuous "
            "latent imagination [2]. DreamerV2 replaced Gaussian stochastic states with multiple categorical "
            "variables and used straight-through gradients, reaching human-level Atari performance [3]. "
            "DreamerV3 retains discrete RSSM states but targets one fixed configuration across very different "
            "domains. Its main robustness devices are: separate stop-gradient KL objectives for dynamics and "
            "representation learning; one-nat free bits; a 1% uniform mixture that prevents near-deterministic "
            "categoricals; symlog transforms for wide-range signed targets; two-hot categorical reward/return "
            "prediction; and percentile-based return normalization. Ablations attribute performance to the "
            "combination rather than one trick. The authors report fixed-hyperparameter results on more than "
            "150 tasks spanning Control Suite, Atari, ProcGen, DMLab, Minecraft, and non-visual domains [1].",
            B,
        ),
        p(
            "<b>Assessment.</b> The strongest contribution is robustness: a single agent design handles image and "
            "vector observations, sparse and dense rewards, and discrete and continuous actions. Its limitations "
            "are also important. Imagined behavior inherits model bias; reconstruction spends capacity on details "
            "that may not matter for control; training remains compute-intensive; and benchmark success does not "
            "guarantee safe real-world prediction under distribution shift. Public JAX code provides training "
            "configurations and reproducible benchmark instructions, but no small public-weight inference model "
            "is the core artifact because Dreamer learns online for each environment [4].",
            B,
        ),
        Spacer(1, 4),
        p(
            "Sources for Q1: [1] Hafner et al., <i>Mastering Diverse Domains through World Models</i>, "
            "https://arxiv.org/abs/2301.04104. [2] Hafner et al., <i>Dream to Control</i>, "
            "https://arxiv.org/abs/1912.01603. [3] Hafner et al., <i>Mastering Atari with Discrete World "
            "Models</i>, https://arxiv.org/abs/2010.02193. [4] Official implementation, "
            "https://github.com/danijar/dreamerv3.",
            S,
        ),
        PageBreak(),
        p("Question 2 - Model and Control the Cart-Pole", H1),
        p("A. Linearization about the upright equilibrium", H2),
        p(
            "The simulator state is <i>s</i> = [x, x-dot, theta-dot, theta]<super>T</super>. Define the local "
            "angle phi = theta - pi and deviation state delta = [x, x-dot, theta-dot, phi]<super>T</super>. "
            "Near theta = pi, sin(theta) is approximately -phi, cos(theta) is approximately -1, "
            "cos(theta)<super>2</super> is approximately 1, and products such as theta-dot<super>2</super> "
            "sin(theta) are second order and discarded. Let D = 4(M+m) - 3m.",
            B,
        ),
        p(
            "x-double-dot = [3m g phi - 4b x-dot + 4u] / D<br/>"
            "theta-double-dot = [6(M+m)g phi - 6b x-dot + 6u] / (lD)",
            EQ,
        ),
        p(
            "Therefore delta-dot = A delta + B u, with the state order used by the code:", B
        ),
        matrix_table(
            [
                ["A =", "[ 0       1       0          0              ]", "B =", "[ 0       ]"],
                ["", "[ 0    -4b/D      0       3mg/D             ]", "", "[ 4/D     ]"],
                ["", "[ 0  -6b/(lD)     0   6(M+m)g/(lD)        ]", "", "[ 6/(lD) ]"],
                ["", "[ 0       0       1          0              ]", "", "[ 0       ]"],
            ],
            [0.36 * inch, 3.55 * inch, 0.34 * inch, 1.0 * inch],
        ),
        Spacer(1, 6),
        p(
            "With M=m=l=0.5, g=9.82, D=2.5. The handout states b=0.1, giving A[2,2]=-0.16 "
            "and A[3,2]=-0.48. Inspection of the supplied executable CartPole class shows b=1.0, so the "
            "implemented model uses the following values; all other entries agree:",
            B,
        ),
        matrix_table(
            [
                ["A =", "[ 0    1.0    0    0      ]", "B =", "[ 0   ]"],
                ["", "[ 0   -1.6    0    5.892  ]", "", "[ 1.6 ]"],
                ["", "[ 0   -4.8    0   47.136  ]", "", "[ 4.8 ]"],
                ["", "[ 0    0.0    1    0      ]", "", "[ 0   ]"],
            ],
            [0.36 * inch, 2.9 * inch, 0.34 * inch, 0.8 * inch],
        ),
        p("B. LQR design and stability", H2),
        p(
            "I used Q = diag(2, 1, 2, 120) and R = [0.2]. The continuous algebraic Riccati equation "
            "returns K = [-3.1623, -6.4776, 7.2892, 47.0906]. The code applies u = K(g-s), with the "
            "angular component wrapped to [-pi, pi). Equivalently, u = -K delta. The closed-loop poles are "
            "-16.7176, -7.7261, and -0.8903 +/- 0.6012i, so the linear model is asymptotically stable. "
            "For comparability with swing-up, force is limited to +/-30 N.",
            B,
        ),
        PageBreak(),
        p("Balancing experiments", H1),
        p(
            "Each deterministic run used the nonlinear supplied ODE simulator for 20 s at dt=0.005 s. "
            "Success means that throughout the final 5 s, |theta-pi|<0.10 rad and |x|<0.50 m. "
            "Capture means |theta-pi|<0.10 rad and |theta-dot|<0.25 rad/s. "
            "The same Q, R, K, and force limit were used for every trial.",
            B,
        ),
    ]

    with RESULTS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    balance = [row for row in rows
               if row["controller"] == "lqr" and float(row["offset_rad"]) != math.pi]
    swingup = next(row for row in rows if row["controller"] == "hybrid")
    table_data = [["initial offset", "capture time (s)", "max |u| (N)", "final-window max |angle|", "result"]]
    names = ["default (pi/40)", "0.01", "0.10", "pi/8", "pi/4"]
    for name, row in zip(names, balance):
        table_data.append(
            [
                name,
                f"{float(row['capture_time_s']):.3f}",
                f"{float(row['max_force_N']):.2f}",
                f"{float(row['tail_max_angle_error_rad']):.2e}",
                "PASS" if row["success"] == "True" else "FAIL",
            ]
        )
    results_table = Table(table_data, colWidths=[0.92*inch, 1.12*inch, 0.92*inch, 1.62*inch, 0.62*inch], repeatRows=1)
    results_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "BodyBold"),
        ("FONTNAME", (0,1), (-1,-1), "Body"),
        ("FONTSIZE", (0,0), (-1,-1), 8.4),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173f73")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [
        results_table,
        Spacer(1, 8),
        p(
            "The default start and all four requested offsets balance. The pi/4 case hits the 30 N force "
            "limit and takes 1.205 s to meet the capture criterion; smaller tested errors remain unsaturated. "
            "The largest successful tested offset is pi/4 (0.785 rad), with all other initial state "
            "deviations zero. These sampled results do not certify a continuous range or global stability: "
            "the linearization and Riccati proof are local, and saturation alters the nonlinear closed loop.",
            B,
        ),
        p("C. Swing-up from the downward configuration", H2),
        p(
            "With initial-offset=pi, the simulator begins at theta=2pi, physically downward. Pure LQR performs "
            "poorly: the local angle error is maximally ambiguous at +/-pi, the command saturates, and the pole "
            "does not settle upright during 20 s. Its final-window maximum angle error is 3.1411 rad. Tuning Q "
            "and R changes local aggressiveness but does not give LQR a global energy-building strategy.",
            B,
        ),
        p(
            "I therefore explored a hybrid controller. Away from upright it shapes the pole energy<br/>"
            "E = (m l<super>2</super>/6) theta-dot<super>2</super> + (mgl/2)(1-cos(theta)), "
            "with target E* = mgl, using u = -k<sub>E</sub>(E*-E) sign(theta-dot cos(theta)) "
            "- k<sub>x</sub>x - k<sub>v</sub>x-dot, clipped to +/-30 N. I used k<sub>E</sub>=40, "
            "k<sub>x</sub>=1, and k<sub>v</sub>=2. When |theta-pi|<0.42 rad and |theta-dot|<3.5 rad/s, "
            "control switches to LQR; 0.65 rad hysteresis prevents chatter.",
            B,
        ),
        Image(str(PLOT), width=6.25 * inch, height=2.97 * inch),
        p(
            "Figure 1. Downward-start comparison. Upright score cos(theta-pi) is +1 upright and -1 "
            "downward. Energy shaping swings up the pole; LQR then recenters the cart.",
            S,
        ),
        PageBreak(),
        p("Single-pole swing-up result", H1),
        p(
            "The hybrid controller first meets |theta-pi|<0.10 rad and |theta-dot|<0.25 rad/s at 1.340 s. "
            "It passes the 20 s success criterion: over the final 5 s, maximum angle error is "
            f"{float(swingup['tail_max_angle_error_rad']):.2e} rad and maximum |x| is "
            f"{float(swingup['tail_max_cart_position_m']):.2e} m. Peak force is 30 N. This is my best "
            "swing-up result. It is stronger than pure LQR but should be interpreted as a deterministic "
            "simulation result; actuator delay, sensor noise, and model mismatch were not tested.",
            B,
        ),
        p("D. DoubleCartpole bonus", H1),
        p(
            "For the six-state system [x, x-dot, theta1-dot, theta2-dot, theta1, theta2], "
            "let phi1 = theta1 - pi and phi2 = theta2 - pi. At the upright equilibrium, "
            "the simulator's coupled acceleration equations simplify to H0 a = r, with "
            "a = [x-double-dot, theta1-double-dot, theta2-double-dot]<super>T</super>:",
            B,
        ),
        p(
            "H0 = [[3, 0.9, 0.3], [4.5, 2.4, 0.9], [3, 1.8, 1.2]]<br/>"
            "r = [-0.2 x-dot + 2u, 44.19 phi1, 29.46 phi2]<super>T</super>",
            EQ,
        ),
        p(
            "Substituting a = H0<super>-1</super>r into the kinematic equations gives "
            "delta-dot = A delta + Bu. With Q = diag(2, 1, 2, 2, 120, 120), R = [0.2], "
            "the continuous LQR gain is K = [3.1623, 6.8993, 0.8285, 27.1835, -135.5356, "
            "189.0925]. All six eigenvalues of A-BK have negative real part. A central-difference "
            "check against the nonlinear simulator agrees with the derived A and B to about 1e-8.",
            B,
        ),
    ]
    with DOUBLE_RESULTS.open(newline="") as stream:
        double_rows = list(csv.DictReader(stream))
    double_labels = ["default", "0.01", "0.10", "pi/8", "0.42", "pi/4", "pi"]
    double_table_data = [["start offset", "capture (s)", "tail max angle (rad)",
                          "peak |x| (m)", "result"]]
    for label, row in zip(double_labels, double_rows):
        capture = row["capture_time_s"]
        double_table_data.append([
            label,
            f"{float(capture):.2f}" if capture else "--",
            f"{max(float(row['tail_max_pole1_error_rad']), float(row['tail_max_pole2_error_rad'])):.3g}",
            f"{float(row['max_cart_position_m']):.3g}",
            "PASS" if row["success"] == "True" else "FAIL",
        ])
    double_table = Table(double_table_data,
                         colWidths=[0.85*inch, 0.85*inch, 1.45*inch,
                                    1.05*inch, 0.65*inch], repeatRows=1)
    double_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "BodyBold"),
        ("FONTNAME", (0,1), (-1,-1), "Body"),
        ("FONTSIZE", (0,0), (-1,-1), 8.4),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173f73")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [
        double_table,
        Spacer(1, 5),
        p(
            "Both poles balance from the default start through pi/8 with a 40 N cap. "
            "The pi/8 trial briefly moves the cart 1.23 m, then recenters it; "
            "0.42 rad, pi/4, and the downward start do not recover. A 30 N cap also failed at "
            "pi/8. The largest successful tested offset is 0.393 rad, and the next tested offset "
            "(0.42 rad) fails; no untested initial conditions are claimed.",
            B,
        ),
        Image(str(DOUBLE_PLOT), width=6.25 * inch, height=2.42 * inch),
        p("Figure 2. Both pole angles and cart position during the pi/8 recovery.", S),
        PageBreak(),
        p("E. Research bonus - learned world model", H1),
        p(
            "I collected 7,680 training and 1,920 held-out transitions with randomized initial states "
            "and forces, storing each row in the simulator's (next state, state, force) format. A compact "
            "action-conditioned model fits trigonometric features of the two accelerations by least squares. "
            "Each output has the learned form acceleration = feature numerator / "
            "(1 - a cos<super>2</super>(theta)); fourth-order integration predicts the next state. "
            "The fit uses transition data and no simulator constants.",
            B,
        ),
    ]
    with BONUS.open(newline="") as stream:
        bonus = dict(csv.reader(stream))
    story += [
        p(
            "On 12 held-out 0.8 s rollouts driven by recorded forces, root mean square error is "
            f"{float(bonus['heldout_open_loop_0p8s_rmse_x']):.2e} m in cart position and "
            f"{float(bonus['heldout_open_loop_0p8s_rmse_theta']):.2e} rad in pole angle. "
            "A 50 ms receding-horizon controller evaluates candidate forces with the learned model, "
            "a quadratic running cost, and an LQR terminal value. It balances offsets 0.1, pi/8, "
            "and pi/4 in separate 8 s simulator trials; final 2 s angle errors stay below 0.0037 rad. "
            "This demonstrates model use in control, though the planner is not shown to outperform LQR "
            "or swing up from downward.",
            B,
        ),
        p("Held-out prediction and control", H2),
    ]
    error_table = Table([
        ["state variable", "one-step RMSE", "0.8 s open-loop RMSE"],
        ["cart position (m)",
         f"{float(bonus['heldout_one_step_rmse_x']):.2e}",
         f"{float(bonus['heldout_open_loop_0p8s_rmse_x']):.2e}"],
        ["pole angle (rad)",
         f"{float(bonus['heldout_one_step_rmse_theta']):.2e}",
         f"{float(bonus['heldout_open_loop_0p8s_rmse_theta']):.2e}"],
    ], colWidths=[1.65*inch, 1.55*inch, 1.75*inch])
    error_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "BodyBold"),
        ("FONTNAME", (0,1), (-1,-1), "Body"),
        ("FONTSIZE", (0,0), (-1,-1), 8.4),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173f73")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    control_table = Table([
        ["start offset", "tail max angle (rad)", "tail max |x| (m)", "peak |u| (N)"],
        *[
            [label,
             f"{float(bonus[f'lookahead_offset_{offset:.6f}_tail_max_angle_error_rad']):.3g}",
             f"{float(bonus[f'lookahead_offset_{offset:.6f}_tail_max_cart_position_m']):.3g}",
             f"{float(bonus[f'lookahead_offset_{offset:.6f}_peak_force_N']):.1f}"]
            for label, offset in (("0.1", 0.1), ("pi/8", math.pi/8),
                                  ("pi/4", math.pi/4))
        ],
    ], colWidths=[1.0*inch, 1.5*inch, 1.25*inch, 1.2*inch])
    control_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "BodyBold"),
        ("FONTNAME", (0,1), (-1,-1), "Body"),
        ("FONTSIZE", (0,0), (-1,-1), 8.4),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173f73")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [
        error_table,
        Spacer(1, 7),
        control_table,
        Spacer(1, 7),
        Image(str(MODEL_PLOT), width=6.25 * inch, height=2.38 * inch),
        p(
            "Figure 3. One unseen trajectory: predicted pole rate follows the simulator; lower panel "
            "shows accumulated angle prediction error in milliradians.",
            S,
        ),
        p(
            "Reproduce all results from <b>assignment1/TeachingCartpole</b> with "
            "<b>python3 experiment.py</b>, <b>python3 double_experiment.py</b>, "
            "and <b>python3 world_model_bonus.py</b>. "
            "The simulator source is D. Meger et al., "
            "https://github.com/dmeger/TeachingCartpole.",
            S,
        ),
    ]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
