"""Three-seed UDR/RGB evaluation; optionally add historical v3.0 GRS and v3.1 GTSS.

V3.0 weights are deliberately left as a placeholder until final comparison.
Each model runs in a separate process using the same seed list and test pipeline.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(runs, seeds):
    """Summarize measured runs only; all deltas use the RGB baseline."""
    report = dict(seeds=seeds, std_ddof=1, runs=runs, summary={})
    lines = ['# UDR-MambaSR paired evaluation', '',
             'Three matched seeds; sample standard deviation (ddof=1).', '',
             '| Dataset | Model | PSNR mean +/- std | SSIM mean +/- std | Delta PSNR vs RGB |',
             '|---|---|---:|---:|---:|']
    for dataset in runs['baseline'][0]:
        summary = {}
        for label, measurements in runs.items():
            summary[label] = {metric: dict(mean=statistics.mean(run[dataset][metric] for run in measurements),
                                           std=statistics.stdev(run[dataset][metric] for run in measurements))
                              for metric in ('psnr', 'ssim')}
        for label, values in summary.items():
            values['delta_psnr_vs_rgb'] = values['psnr']['mean'] - summary['baseline']['psnr']['mean']
            psnr, ssim = values['psnr'], values['ssim']
            lines.append(f"| {dataset} | {label} | {psnr['mean']:.4f} +/- {psnr['std']:.4f} | {ssim['mean']:.4f} +/- {ssim['std']:.4f} | {values['delta_psnr_vs_rgb']:+.4f} |")
        report['summary'][dataset] = summary
    report['average_delta_vs_rgb'] = {label: statistics.mean(
        ds[label]['delta_psnr_vs_rgb'] for ds in report['summary'].values()) for label in runs}
    def delta(dataset):
        return report['summary'][dataset]['udr']['delta_psnr_vs_rgb']
    avg = report['average_delta_vs_rgb']['udr']
    report['udr_criteria'] = dict(average_positive=avg > 0, average_target_met=avg >= 0.05,
        urban100_target_met=delta('Urban100') >= 0.05,
        manga109_target_met=delta('Manga109') >= 0.05, b100_delta_psnr=delta('B100'))
    paired = {ds: [u[ds]['psnr'] - b[ds]['psnr'] for u, b in zip(runs['udr'], runs['baseline'])]
              for ds in report['summary']}
    report['paired_delta_psnr'] = {ds: dict(values=v, mean=statistics.mean(v), std=statistics.stdev(v)) for ds, v in paired.items()}
    stable = {ds: min(paired[ds]) > 0 for ds in ('Urban100', 'Manga109')}
    report['udr_criteria']['stable_positive_all_three_seeds'] = stable
    report['udr_criteria']['stop_rule_triggered'] = avg <= 0 and not any(stable.values())
    lines.extend(['', f'UDR five-set average delta vs RGB: {avg:+.4f} dB.',
                  'Criteria: '+json.dumps(report['udr_criteria'], sort_keys=True)])
    return report, '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-config', default='options/test/mambairv2/test_UDR_RGB_reference_x4.yml')
    p.add_argument('--udr-config', default='options/test/mambairv2/test_UDR_MambaSR_x4.yml')
    p.add_argument('--grs-config', default='options/test/mambairv2/test_GTSS_GRS_reference_x4.yml')
    p.add_argument('--baseline-checkpoint')
    p.add_argument('--udr-checkpoint')
    p.add_argument('--grs-checkpoint', help='Fill after UDR training, before the historical-model comparison.')
    p.add_argument('--gtss-config', default='options/test/mambairv2/test_GTSS_MambaSR_x4.yml')
    p.add_argument('--gtss-checkpoint', help='Optional historical v3.1 full checkpoint.')
    p.add_argument('--include-gtss', action='store_true')
    p.add_argument('--include-grs', action='store_true', help='Also evaluate the historical v3.0 model.')
    p.add_argument('--seeds', nargs=3, type=int, default=[10,11,12])
    p.add_argument('--output', default='results/udr_comparison')
    p.add_argument('--worker', nargs=2, metavar=('CONFIG','JSON'))
    a = p.parse_args()
    if a.worker:
        # Retain the tested v3.0 worker: sorted filenames, seed reset AFTER
        # construction and before each dataset, same cuDNN and partition settings.
        from scripts.grs.evaluate_repeated import worker
        worker(*a.worker)
        return
    if len(set(a.seeds)) != 3:
        p.error('Use three distinct seeds paired identically between models.')
    labels = ['baseline','udr']
    if a.include_grs or a.grs_checkpoint:
        labels.append('grs')
    if a.include_gtss or a.gtss_checkpoint:
        labels.append('gtss')
    configs = {}
    for label in labels:
        cfgpath = Path(getattr(a, label+'_config'))
        if not cfgpath.is_absolute():
            cfgpath = ROOT/cfgpath
        cfg = yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
        ckpt = getattr(a, label+'_checkpoint') or cfg['path'].get('pretrain_network_g')
        if not ckpt or str(ckpt).startswith('__SET_'):
            p.error(f'{label}: checkpoint is reserved. AFTER UDR training, fill '
                    f'{cfgpath}: path.pretrain_network_g or pass --{label}-checkpoint PATH '
                    'before the historical-model comparison. Training and UDR-only testing need no GRS weights.')
        ckpt = Path(ckpt).expanduser()
        if not ckpt.is_absolute():
            ckpt = ROOT/ckpt
        cfg['path']['pretrain_network_g'] = str(ckpt.resolve())
        cfg['path']['strict_load_g'] = True
        configs[label] = cfg
    for label, cfg in configs.items():
        if not Path(cfg['path']['pretrain_network_g']).is_file():
            p.error(f"{label}: checkpoint not found: {cfg['path']['pretrain_network_g']}")
        ref = configs['baseline']
        if cfg['val']['metrics'] != ref['val']['metrics'] or cfg['scale'] != ref['scale']:
            p.error(f'{label}: evaluation metric/scale settings differ from RGB.')
        if set(cfg['datasets']) != set(ref['datasets']):
            p.error(f'{label}: mismatched dataset list.')
        for key, ds in ref['datasets'].items():
            for field in ('name','dataroot_gt','dataroot_lq','filename_tmpl'):
                if cfg['datasets'][key][field] != ds[field]:
                    p.error(f'{label}: unmatched {key}/{field}.')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = {label: [] for label in labels}
    for seed in a.seeds:
        for label, cfg in configs.items():
            config = copy.deepcopy(cfg)
            config['manual_seed'] = seed
            config['name'] = f'UDR_fair_{label}_seed{seed}'
            cfgpath, result = out/f'{label}_{seed}.yml', out/f'{label}_{seed}.json'
            cfgpath.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                            str(cfgpath), str(result)], cwd=ROOT, check=True)
            runs[label].append(json.loads(result.read_text(encoding='utf-8')))
    report, markdown = summarize(runs, a.seeds)
    report['checkpoints'] = {label: cfg['path']['pretrain_network_g'] for label,cfg in configs.items()}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'summary.md').write_text(markdown,encoding='utf-8')
    print(markdown)


if __name__ == '__main__':
    main()
