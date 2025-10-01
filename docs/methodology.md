## Methodology (overview)
- Stability pre-filter: Rosetta cartesian_relax + cartesian_ddg.
- Active-site aware design: Rosetta FastDesign with catalytic constraints.
- Cross-check: FoldX BuildModel on top candidates.
- Structure prediction (optional): ColabFold/AlphaFold for novel backbones.
- Backbone generation (optional): RFdiffusion; post-relax with Rosetta; select by scores.
