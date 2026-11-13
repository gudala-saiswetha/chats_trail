import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_diagram():
    # High-resolution figure (16:9 widescreen)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color Palette - Clean Academic Style
    C_BG = '#FFFFFF'
    C_OUTER_STAGE_1 = '#F8FAFC'
    C_OUTER_STAGE_2 = '#EFF6FF'
    C_OUTER_STAGE_3 = '#F8FAFC'
    C_OUTER_STAGE_4 = '#F1F5F9'

    BORDER_STAGE = '#334155'
    BORDER_MODEL = '#2563EB'
    BORDER_BOX = '#64748B'

    COLOR_TITLE = '#0F172A'
    COLOR_SUBTITLE = '#475569'
    COLOR_ARROW = '#1E293B'

    def draw_rounded_box(x, y, w, h, title='', subtitle='', bg='#FFFFFF', border='#64748B', lw=1.5, fontsize=10, bold=True, text_color='#0F172A'):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.18',
                             linewidth=lw, edgecolor=border, facecolor=bg, zorder=2)
        ax.add_patch(box)
        if title and subtitle:
            ax.text(x + w/2, y + h/2 + 0.14, title, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold' if bold else 'normal', color=text_color, zorder=3)
            ax.text(x + w/2, y + h/2 - 0.16, subtitle, ha='center', va='center',
                    fontsize=fontsize * 0.82, color=COLOR_SUBTITLE, zorder=3)
        elif title:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold' if bold else 'normal', color=text_color, zorder=3)
        return box

    # ---------------------------------------------------------------------------
    # HEADER / TITLE
    # ---------------------------------------------------------------------------
    ax.text(8.0, 8.55, 'MLS-SHIELD: System Architecture Diagram',
            ha='center', va='center', fontsize=21, fontweight='bold', color=COLOR_TITLE)
    ax.text(8.0, 8.20, 'Multi-Task Layered Semantic Shield for LLM Adversarial Prompt Defense',
            ha='center', va='center', fontsize=11.5, color=COLOR_SUBTITLE, fontstyle='italic')

    # ---------------------------------------------------------------------------
    # 1. INPUT & PREPROCESSING STAGE (BOX 1)
    # ---------------------------------------------------------------------------
    b1 = FancyBboxPatch((0.5, 2.2), 3.0, 5.6, boxstyle='round,pad=0.03,rounding_size=0.22',
                        linewidth=2.0, edgecolor=BORDER_STAGE, facecolor=C_OUTER_STAGE_1, zorder=1)
    ax.add_patch(b1)
    ax.text(2.0, 7.42, '1. INPUT &\nPREPROCESSING STAGE', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_TITLE, zorder=3)

    draw_rounded_box(0.75, 5.85, 2.5, 0.95, 'Raw User Query', '(Unscreened Prompt Text)',
                     bg='#FFFFFF', border=BORDER_BOX, fontsize=10.5)

    draw_rounded_box(0.75, 4.30, 2.5, 0.95, 'Text Sanitization', '& Normalization',
                     bg='#FFFFFF', border=BORDER_BOX, fontsize=10.5)

    draw_rounded_box(0.75, 2.50, 2.5, 1.25, 'Head-Tail Slicing', 'Preserves First 120 &\nLast 120 Words (240 total)',
                     bg='#FFFFFF', border=BORDER_BOX, fontsize=9.8)

    # Vertical connectors in Stage 1
    ax.annotate('', xy=(2.0, 5.25), xytext=(2.0, 5.85), arrowprops=dict(arrowstyle='->', lw=2.0, color=COLOR_ARROW))
    ax.annotate('', xy=(2.0, 3.75), xytext=(2.0, 4.30), arrowprops=dict(arrowstyle='->', lw=2.0, color=COLOR_ARROW))

    # ---------------------------------------------------------------------------
    # 2. PROPOSED MODEL: MLS-SHIELD (BOX 2)
    # ---------------------------------------------------------------------------
    b2 = FancyBboxPatch((3.95, 2.2), 7.85, 5.6, boxstyle='round,pad=0.03,rounding_size=0.22',
                        linewidth=2.2, edgecolor=BORDER_MODEL, facecolor=C_OUTER_STAGE_2, zorder=1)
    ax.add_patch(b2)
    ax.text(7.87, 7.46, 'PROPOSED MODEL: MLS-SHIELD', ha='center', va='center',
            fontsize=13.5, fontweight='bold', color=BORDER_MODEL, zorder=3)
    ax.text(7.87, 7.16, '(Multi-Task Layered Semantic Shield)', ha='center', va='center',
            fontsize=10.5, color=COLOR_SUBTITLE, zorder=3)

    # Connector from Stage 1 to Stage 2
    ax.annotate('', xy=(3.95, 4.70), xytext=(3.50, 4.70), arrowprops=dict(arrowstyle='->', lw=2.4, color=COLOR_ARROW))

    # Branch A: Transformer Semantic Encoder
    draw_rounded_box(4.25, 5.25, 2.45, 1.45, 'Transformer\nSemantic Encoder', '(DistilRoBERTa, 768-dim)',
                     bg='#FFFFFF', border='#2563EB', lw=1.8, fontsize=10.5)

    # Branch B: 14 Handcrafted Pattern Branch (Typo fixed!)
    draw_rounded_box(4.25, 2.70, 2.45, 1.45, '14 Handcrafted\nPattern Branch', '(14-dim -> 64-dim MLP)',
                     bg='#FFFFFF', border='#7C3AED', lw=1.8, fontsize=10.5)

    # Gated Cross-Feature Fusion Layer
    draw_rounded_box(7.15, 3.35, 1.45, 2.70, 'Gated\nFusion\nLayer', '(Sigmoid Gate\n256-dim)',
                     bg='#F0FDF4', border='#16A34A', lw=2.0, fontsize=10.5)

    # Connectors to Gated Fusion
    ax.annotate('', xy=(7.15, 5.00), xytext=(6.70, 5.70), arrowprops=dict(arrowstyle='->', lw=2.0, color='#2563EB'))
    ax.annotate('', xy=(7.15, 4.40), xytext=(6.70, 3.70), arrowprops=dict(arrowstyle='->', lw=2.0, color='#7C3AED'))

    # Multi-Task Prediction Heads Box
    draw_rounded_box(9.05, 2.70, 2.45, 4.00, 'Multi-Task\nPrediction Heads',
                     '• Binary Head\n  (2 classes: Safe/Atk)\n\n• Attack Category\n  (4 classes)\n\n• Risk Severity\n  (5 ordered tiers)',
                     bg='#FFFFFF', border='#4F46E5', lw=1.8, fontsize=10.2)

    # Connector Fusion -> Prediction Heads
    ax.annotate('', xy=(9.05, 4.70), xytext=(8.60, 4.70), arrowprops=dict(arrowstyle='->', lw=2.4, color='#16A34A'))

    # ---------------------------------------------------------------------------
    # 3. SECURITY DECISION ENGINE (BOX 3)
    # ---------------------------------------------------------------------------
    b3 = FancyBboxPatch((12.25, 2.2), 3.25, 5.6, boxstyle='round,pad=0.03,rounding_size=0.22',
                        linewidth=2.0, edgecolor=BORDER_STAGE, facecolor=C_OUTER_STAGE_3, zorder=1)
    ax.add_patch(b3)
    ax.text(13.87, 7.42, '3. SECURITY\nDECISION ENGINE', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_TITLE, zorder=3)

    # Connector Model -> Decision Engine
    ax.annotate('', xy=(12.25, 4.70), xytext=(11.50, 4.70), arrowprops=dict(arrowstyle='->', lw=2.4, color=COLOR_ARROW))

    # Decisions: ALLOW, FLAG, BLOCK
    draw_rounded_box(12.50, 5.75, 2.75, 1.00, 'ALLOW', 'Safe -> Forwarded to Target LLM',
                     bg='#ECFDF5', border='#059669', lw=2.0, fontsize=11, text_color='#065F46')

    draw_rounded_box(12.50, 4.25, 2.75, 1.00, 'FLAG', 'Borderline -> Quarantine Review',
                     bg='#FEFCE8', border='#CA8A04', lw=2.0, fontsize=11, text_color='#854D0E')

    draw_rounded_box(12.50, 2.75, 2.75, 1.00, 'BLOCK', 'Malicious -> Halt & Log Attack',
                     bg='#FEF2F2', border='#DC2626', lw=2.0, fontsize=11, text_color='#991B1B')

    # ---------------------------------------------------------------------------
    # 4. PERFORMANCE EVALUATION (BOX 4)
    # ---------------------------------------------------------------------------
    b4 = FancyBboxPatch((0.5, 0.40), 15.0, 1.45, boxstyle='round,pad=0.03,rounding_size=0.20',
                        linewidth=1.8, edgecolor='#94A3B8', facecolor=C_OUTER_STAGE_4, zorder=1)
    ax.add_patch(b4)

    ax.text(8.0, 1.35, '4. PERFORMANCE EVALUATION',
            ha='center', va='center', fontsize=13, fontweight='bold', color=COLOR_TITLE, zorder=3)

    metrics_text = 'Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), Confusion Matrix, Inference Latency'
    ax.text(8.0, 0.82, metrics_text, ha='center', va='center', fontsize=11.0, color='#334155', zorder=3)

    # Connector Decision Engine -> Performance
    ax.annotate('', xy=(13.87, 1.85), xytext=(13.87, 2.20), arrowprops=dict(arrowstyle='->', lw=2.0, color=COLOR_ARROW))

    # ---------------------------------------------------------------------------
    # SAVE FIGURE (HIGH RESOLUTION 300 DPI)
    # ---------------------------------------------------------------------------
    os.makedirs('docs', exist_ok=True)
    out_1 = 'docs/mls_shield_architecture_diagram.png'
    out_2 = '/Users/vsskarthikgudala/.gemini/antigravity/brain/face7253-b9af-44d3-96dd-240a0a7e7d2d/mls_shield_architecture_diagram.png'

    plt.savefig(out_1, dpi=300, bbox_inches='tight')
    plt.savefig(out_2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ High-resolution corrected diagram created successfully at:\n  - {out_1}\n  - {out_2}")

if __name__ == '__main__':
    create_diagram()

