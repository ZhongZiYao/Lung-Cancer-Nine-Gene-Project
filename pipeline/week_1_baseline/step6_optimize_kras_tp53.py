# -*- coding: utf-8 -*-
"""step6_optimize_kras_tp53.py — Optimize KRAS/TP53 with class-balanced strategies"""
import subprocess
import sys
import time
from pathlib import Path

# KRAS and TP53 strategies
# Both have heavy imbalance
# Try different pos_weight_max and lr
TARGETS = [
    # (gene, pos_w_max, lr, hidden, dropout, epochs, patience, tag)
    ('KRAS', 30.0, 5e-4, 64, 0.5, 20, 8, 'pw30_lr5e4_h64'),
    ('KRAS', 50.0, 5e-4, 64, 0.5, 20, 8, 'pw50_lr5e4_h64'),
    ('KRAS', 20.0, 3e-4, 64, 0.5, 20, 8, 'pw20_lr3e4_h64'),
    ('TP53', 5.0, 5e-4, 64, 0.5, 20, 8, 'pw5_lr5e4_h64'),
    ('TP53', 10.0, 5e-4, 64, 0.5, 20, 8, 'pw10_lr5e4_h64'),
    ('TP53', 20.0, 3e-4, 64, 0.5, 20, 8, 'pw20_lr3e4_h64'),
]

PY = r'D:\conda\envs\lung9gene\python.exe'
SCRIPT = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\step4_iter_train.py'
FEAT_DIR = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'
OUT_BASE = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter5_optimize'
LOG_FILE = Path(r'D:\tmp\iter5_optimize.log')

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    s = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(s, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(s + '\n')

if __name__ == '__main__':
    Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
    log('=== iter5_optimize START, %d runs ===' % len(TARGETS))
    for gene, pw, lr, hidden, dropout, epochs, patience, tag in TARGETS:
        out_dir = Path(OUT_BASE) / ('%s_%s' % (gene, tag))
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = Path(r'D:\tmp') / ('iter5_%s_%s_stdout.log' % (gene, tag))
        stderr_log = Path(r'D:\tmp') / ('iter5_%s_%s_stderr.log' % (gene, tag))
        cmd = [PY, SCRIPT,
               '--features-dir', FEAT_DIR,
               '--gene', gene,
               '--cv', '3',
               '--epochs', str(epochs),
               '--patience', str(patience),
               '--hidden-dim', str(hidden),
               '--dropout', str(dropout),
               '--lr', str(lr),
               '--pos-w-max', str(pw),
               '--n-seeds', '2',
               '--output-dir', str(out_dir)]
        log('=== %s %s ===' % (gene, tag))
        log('CMD: %s' % ' '.join(cmd))
        t0 = time.time()
        with open(stdout_log, 'w') as sout, open(stderr_log, 'w') as serr:
            r = subprocess.run(cmd, stdout=sout, stderr=serr, timeout=14400)
        elapsed = time.time() - t0
        log('  %s %s rc=%d (%.0fs)' % (gene, tag, r.returncode, elapsed))
        summary_file = out_dir / (gene + '_3') / 'single_summary.json'
        if summary_file.exists():
            import json
            with open(summary_file) as f:
                s = json.load(f)
            m = s.get('mean', {})
            std = s.get('std', {})
            log('  AUC=%.3f±%.3f ACC=%.3f F1=%.3f' % (
                m.get('auc', 0), std.get('auc', 0), m.get('acc', 0), m.get('f1', 0)))
        else:
            log('  WARN: no summary')
    log('=== iter5_optimize END ===')