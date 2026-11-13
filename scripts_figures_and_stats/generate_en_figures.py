# -*- coding: utf-8 -*-
r"""
Unified, verified entry-point for regenerating JMR submission figures.
Delegates directly to the audited key-bound figure generators in
04_审阅核验链/22_Gemini独立审计_20261106/ so that legacy hardcoded arrays or
external workspace paths can never overwrite the validated manuscript figures.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106"
sys.path.insert(0, str(AUDIT_DIR))

import regenerate_fig3_clean
import regenerate_fig6_clean
import regenerate_graphical_abstract
import regenerate_supp_figures_clean


def main():
    regenerate_fig3_clean.main()
    regenerate_fig6_clean.render_fig6("EN")
    regenerate_fig6_clean.render_fig6("CN")
    regenerate_supp_figures_clean.render_fig_s1("EN")
    regenerate_supp_figures_clean.render_fig_s1("CN")
    regenerate_supp_figures_clean.render_fig_s2("EN")
    regenerate_supp_figures_clean.render_fig_s2("CN")
    regenerate_graphical_abstract.render_ga()
    print("All verified JMR figures regenerated from audited sources.")


if __name__ == "__main__":
    main()
