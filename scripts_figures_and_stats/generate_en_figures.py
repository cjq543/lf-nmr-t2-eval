# -*- coding: utf-8 -*-
"""
Unified, standalone entry-point for regenerating JMR submission figures inside
the public ReproBundle without any external workspace path dependencies.
Outputs are written under results/regenerated_figures/.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

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
    print("All verified JMR figures regenerated standalone under results/regenerated_figures/.")


if __name__ == "__main__":
    main()
