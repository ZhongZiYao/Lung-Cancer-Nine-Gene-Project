# -*- coding: utf-8 -*-
"""
step5_all_genes.py — Train all 9 genes sequentially with best config from iter2 grid
Best: hidden_dim=64, dropout=0.5, lr=5e-4, 3-fold, 20 epoch, patience 8
"""
import subprocess
import sys
import time
from pathlib import Path

GENES = ['EGFR', 'KRAS', 'TP53', 'ALK', 'ROS1', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']
PY = r'D:\conda\envs\lung9gene\python.exe'
SCRIPT = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\step4_iter_train.py'
FEAT_DIR = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'
OUT_BASE = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter4_all'
LOG_DIR = r'D:\tmp'

LOG_FILE = Path(LOG_DIR) / 'iter4_all.log'
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    s = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(s, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(s + '\n')

if __name__ == '__main__':
    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    log('=== iter4_all START, %d genes ===' % len(GENES))
    for gene in GENES:
        out_dir = Path(OUT_BASE) / ('%s_3fold' % gene)
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = Path(LOG_DIR) / ('iter4_%s_stdout.log' % gene)
        stderr_log = Path(LOG_DIR) / ('iter4_%s_stderr.log' % gene)
        cmd = [PY, SCRIPT,
               '--features-dir', FEAT_DIR,
               '--gene', gene,
               '--cv', '3',
               '--epochs', '20',
               '--patience', '8',
               '--hidden-dim', '64',
               '--dropout', '0.5',
               '--lr', '5e-4',
               '--n-seeds', '2',
               '--output-dir', str(out_dir)]
        log('=== %s ===' % gene)
        log('CMD: %s' % ' '.join(cmd))
        t0 = time.time()
        with open(stdout_log, 'w') as sout, open(stderr_log, 'w') as serr:
            r = subprocess.run(cmd, stdout=sout, stderr=serr, timeout=14400)  # 4h timeout
        elapsed = time.time() - t0
        log('  %s rc=%d (%.0fs)' % (gene, r.returncode, elapsed))
        # Read summary
        summary_file = out_dir / ('%s_%s' % (gene, '3')) / 'single_summary.json'
        if summary_file.exists():
            import json
            with open(summary_file) as f:
                summary = json.load(f)
            log('  AUC=%.3f±%.3f ACC=%.3f F1=%.3f' % (
                summary['mean']['auc'], summary['std']['auc'],
                summary['mean']['acc'], summary['mean']['f1']))
        else:
            log('  WARN: no summary found at %s' % summary_file)
    log('=== iter4_all END ===')