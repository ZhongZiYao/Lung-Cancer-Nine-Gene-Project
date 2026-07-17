"""Inspect DeepGEM pickle checkpoint structure."""
import pickle

ckpts = [
    'checkpoints/DeepGEM/model_ExcisionalBiopsy_EGFR.pickle',
    'checkpoints/DeepGEM/model_AspirationBiopsy_EGFR.pickle',
    'checkpoints/DeepGEM/model_ExcisionalBiopsy_ALK.pickle',
    'checkpoints/DeepGEM_TCGA/modelTCGA_ExcisionalBiopsy_EGFR.pickle',
]
for c in ckpts:
    with open(c, 'rb') as f:
        d = pickle.load(f)
    print('=' * 70)
    print(c)
    print('  keys :', list(d.keys()))
    print('  parameter:', d['parameter'])
    state = d['checkpoint']
    print('  state_dict num tensors:', len(state))
    print('  state_dict layers (前 8):')
    for k, v in list(state.items())[:8]:
        print(f'    {k:50s} {tuple(v.shape) if hasattr(v, "shape") else v}')