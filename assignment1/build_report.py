"""Build the final Assignment 1 PDF from verified experiment outputs."""

from pathlib import Path
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
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
SIM = ROOT / "TeachingCartpole"
sys.path.insert(0, str(SIM))

from cartpole_control import Controller, wrap_to_pi  # noqa: E402
from cartpole_envs import CartPole, DoubleCartPole  # noqa: E402
from q2_report import build_q2_story  # noqa: E402


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
    axes[0].set_ylabel(r"$\cos(\theta-\pi)$")
    axes[0].set_ylim(-1.1, 1.12)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[1].plot(pure[:, 0], pure[:, 2], color="#9a3412", lw=1.2)
    axes[1].plot(hybrid[:, 0], hybrid[:, 2], color="#075985", lw=1.2)
    axes[1].axhline(0, color="#64748b", lw=0.7)
    axes[1].set_ylabel(r"cart $x$ (m)")
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
    axes[0].set_ylabel(r"$\dot{\theta}$ (rad/s)")
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
    axes[0].set_ylabel(r"$|\phi_i|$ (rad)")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[1].plot(double[:, 0], double[:, 1], color="#0f766e", lw=1.3)
    axes[1].axhline(0, color="#64748b", lw=0.7)
    axes[1].set_ylabel(r"cart $x$ (m)")
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
    S = styles["Small"]
    story = []

    story += [
        Spacer(1, 0.08 * inch),
        p("Assignment 1: Intro to World Models and Control", styles["TitleCustom"]),
        p("COMP 765 - Fall 2026", styles["Subtitle"]),
        p("Question 1 - Profile of a World Model: DreamerV3", H1),
        p(
            "I chose DreamerV3 because I find its ability to learn across very different tasks "
            "interesting. Its recurrent state-space model (RSSM) learns a compact representation of "
            "the environment and predicts how it changes in response to actions [1].",
            B,
        ),
        p(
            "<b>Inputs and outputs.</b> For training, Dreamer uses sequences of observations, actions, "
            "rewards, and flags indicating "
            "whether the episode continues. Observations can be RGB images or low-dimensional state "
            "vectors. The RSSM combines a memory state "
            "<i>h</i><sub>t</sub> with categorical latent variables <i>z</i><sub>t</sub>. "
            "The encoder uses the current observation and memory to infer the latent state; the dynamics "
            "model predicts it without seeing that observation. From the combined state, the model "
            "reconstructs the observation and predicts reward and continuation. The actor selects a "
            "discrete or continuous action, while the critic estimates a distribution over future return [1].",
            B,
        ),
        p(
            "<b>Design principles and differences.</b> A central design choice is learning from "
            "imagined trajectories. Dreamer trains its actor and "
            "critic on short trajectories predicted in latent space, using predicted rewards and "
            "bootstrapped lambda-returns to account for rewards beyond the rollout. This avoids having "
            "to generate full images for every policy update. Once trained, the actor selects actions "
            "directly, without searching over action sequences at every step. Its dynamics and latent "
            "representation are learned from interaction data. Unlike MuZero's task-focused model, "
            "it also reconstructs observations [1,2].",
            B,
        ),
        p(
            "<b>Theory and key findings.</b> One useful piece of the theory is the KL loss between "
            "the observation-based latent "
            "distribution and the model's prediction. Separate stop-gradient losses train the predictor "
            "to match the encoded state and encourage that state to be predictable. A one-nat free-bits "
            "threshold stops this penalty from dominating when the distributions already agree well. "
            "V3 also mixes in 1% uniform probability, uses symlog to compress large signed values, "
            "predicts rewards and returns with two-hot distributions, and normalizes returns using "
            "percentiles. Together, these changes make training less sensitive to the task's scale. "
            "The authors report results on over 150 tasks with fixed hyperparameters, including "
            "Control Suite, Atari, and Minecraft [1].",
            B,
        ),
        p(
            "<b>Foundational papers and my assessment.</b> The earlier papers help explain how the "
            "model developed. DreamerV1 used continuous stochastic states and propagated gradients "
            "through imagined trajectories to learn behaviour [2]. DreamerV2 switched to categorical "
            "states with straight-through gradients and reached human-level Atari performance [3]. "
            "V3 keeps the discrete RSSM but focuses on making one configuration work across domains [1]. "
            "The public JAX implementation includes training code and configurations [4]. My main "
            "takeaway is that V3 makes world-model learning more reliable across tasks, but prediction "
            "errors can still mislead the policy and training needs substantial computation. I would "
            "want to check how well its predictions hold up in states outside its training experience.",
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
    ]
    story += build_q2_story(SIM, styles, PLOT, DOUBLE_PLOT, MODEL_PLOT)

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
