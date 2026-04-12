"""Vendored ms-pred ICEBERG inference engine (inference-only subset, PyG version).

Submodules:
  - common: chemistry utilities, constants, fingerprints
  - nn_utils: GNN building blocks, MoleculeGNN, SetTransformerEncoder, etc.
  - magma: MAGMa fragmentation engine
  - dag_data: DAG dataset / tree processing
  - gen_model: FragGNN (autoregressive fragment generation)
  - inten_model: IntenGNN (intensity prediction)
  - joint_model: JointModel (combined generation + intensity)

All imports are lazy — heavy dependencies (torch, torch_geometric, pytorch_lightning)
are only loaded when actually accessed via the top-level convenience aliases.
Submodule-to-submodule imports (e.g. joint_model importing gen_model) go
through normal Python relative imports and bypass this __getattr__.
"""
