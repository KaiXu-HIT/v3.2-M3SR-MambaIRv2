"""UDR contracts: --cpu-reference uses a differentiable reference scan, NOT CUDA kernels.

Default mode uses the installed BasicSR/Mamba CUDA stack. --check-data validates
the actual server datasets as well. No performance numbers are generated here.
"""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.grs.check_grs import load_isolated, check_dataset_locally


def check_configs():
    def read(path):
        return yaml.safe_load((ROOT / path).read_text(encoding='utf-8'))
    train = read('options/train/mambairv2/train_GTSS_MambaSR_x4.yml')
    test = read('options/test/mambairv2/test_GTSS_MambaSR_x4.yml')
    configs = []
    for phase in ('A', 'B'):
        cfg = read(f'options/train/mambairv2/train_UDR_MambaSR_x4_phase{phase}.yml')
        assert cfg['datasets'] == train['datasets'], 'Original dataset options changed'
        assert cfg['train']['total_iter'] == 100000
        assert cfg['train']['optim_g']['lr'] == 1e-4
        assert cfg['train']['scheduler']['milestones'] == []
        assert cfg['train']['pixel_opt'] == train['train']['pixel_opt']
        assert cfg['path']['pretrain_kind'] == ('rgb' if phase == 'A' else 'udr')
        configs.append(cfg)
    assert configs[1]['train']['rgb_lr'] == 1e-5
    cfg = read('options/test/mambairv2/test_UDR_MambaSR_x4.yml')
    assert cfg['datasets'] == test['datasets'] and cfg['val'] == test['val']
    configs.append(cfg)
    print('PASS: original dataset paths/options, L1, 100k+100k, fixed differential LR, five-set metrics')
    return configs


def assert_raises(fn, kind=RuntimeError):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f'Expected {kind.__name__}')


def architecture_checks(base, udr, device):
    kw = dict(img_size=8, embed_dim=12, depths=(1,)*6, num_heads=(3,)*6,
              window_size=4, d_state=2, inner_rank=4, num_tokens=4,
              mlp_ratio=1., upscale=4, upsampler='pixelshuffle')
    baseline = base.MambaIRv2(**kw).to(device).eval()
    model = udr.UDRMambaIRv2(**kw).to(device).eval()
    model.load_rgb_state_dict(baseline.state_dict())
    assert all(layer.assm.geometry_controller is None for stage in model.layers
               for layer in stage.residual_group.layers)
    bad = dict(baseline.state_dict())
    bad.pop('conv_first.weight')
    assert_raises(lambda: model.load_rgb_state_dict(bad))
    bad = dict(baseline.state_dict(), unexpected=torch.zeros(1))
    assert_raises(lambda: model.load_rgb_state_dict(bad))
    bad = dict(baseline.state_dict())
    bad['conv_first.weight'] = bad['conv_first.weight'][:1]
    assert_raises(lambda: model.load_rgb_state_dict(bad))
    assert_raises(lambda: model.load_state_dict(baseline.state_dict(), strict=True))
    rgb = torch.rand(2, 3, 5, 7, device=device)
    depth = torch.rand(2, 1, 5, 7, device=device)
    saved_alpha = model.udr.alpha_raw.detach().clone()
    with torch.no_grad():
        model.udr.alpha_raw.zero_()
        torch.manual_seed(61)
        expected = baseline(rgb)
        torch.manual_seed(61)
        actual = model(rgb, depth)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.udr.alpha_raw.copy_(saved_alpha)
        torch.manual_seed(61)
        initial = model(rgb, depth)
        print('Initial small-model SR RMS difference:', (initial - expected).square().mean().sqrt().item())
        assert (initial - expected).abs().max() < 0.01
    print('PASS: strict RGB/full checkpoint contracts and exact alpha=0 RGB equivalence (odd dimensions, B=2)')

    recorded = []
    handles = [stage.register_forward_hook(lambda m, args, out: recorded.append(out.detach().clone()))
               for stage in model.layers]
    with torch.no_grad():
        torch.manual_seed(71)
        a = model(rgb, depth)
        first = recorded[:]
        recorded.clear()
        torch.manual_seed(71)
        b = model(rgb, torch.zeros_like(depth))
        for x, y in zip(first, recorded):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert not torch.equal(a, b), 'Depth residual must have an effect'
    for handle in handles:
        handle.remove()
    print('PASS: changing depth leaves ALL six RGB stages bitwise unchanged but affects final SR')

    stage_maps, all_route_maps = [], []
    def observe_route(module, args, out):
        p = out.float().softmax(-1)
        all_route_maps.append((-(p * p.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(p.shape[-1], device=device))).reshape(2, 1, 8, 8))
    handles = [layer.assm.route.register_forward_hook(observe_route)
               for stage in model.layers[3:] for layer in stage.residual_group.layers]
    handle = model.udr.register_forward_pre_hook(lambda m, args: stage_maps.append(args[3].detach()))
    with torch.no_grad():
        model(rgb, depth)
    torch.testing.assert_close(stage_maps[0], torch.stack(all_route_maps).mean(0))
    for h in handles + [handle]:
        h.remove()
    # Confirm entropy endpoints independently of sampled hard routing policy.
    assm = model.layers[3].residual_group.layers[0].assm
    layer = model.layers[3].residual_group.layers[0]
    original_weight = assm.route[2].weight.detach().clone()
    original_bias = assm.route[2].bias.detach().clone()
    with torch.no_grad():
        assm.route[2].weight.zero_()
        assm.route[2].bias.zero_()
        collected = []
        assm(torch.rand(2, 16, 12, device=device), (4, 4), layer.embeddingA, ambiguity_collector=collected)
        torch.testing.assert_close(collected[0], torch.ones_like(collected[0]))
        assm.route[2].bias[0] = 1000
        collected = []
        assm(torch.rand(2, 16, 12, device=device), (4, 4), layer.embeddingA, ambiguity_collector=collected)
        assert collected[0].abs().max() < 1e-6
        assm.route[2].weight.copy_(original_weight)
        assm.route[2].bias.copy_(original_bias)
    print('PASS: spatial pre-Gumbel entropy, uniform/certain endpoints, stages 4/5/6 aggregation')

    feature = torch.randn(2, 12, 5, 7, device=device)
    with torch.no_grad():
        out, _ = model.udr(feature, rgb, depth, torch.zeros_like(depth))
        torch.testing.assert_close(out, feature, rtol=0, atol=0)
        assert model.udr.gradient(torch.ones_like(depth)).abs().max() == 0
        for value in (-20., 0., 20.):
            model.udr.alpha_raw.fill_(value)
            assert abs(model.udr.alpha.item()) <= 0.100001
        model.udr.alpha_raw.copy_(saved_alpha)
    class ZeroConfidence(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[:, :1])
    gre = model.udr.gre
    model.udr.gre = ZeroConfidence()
    out, _ = model.udr(feature, rgb, depth, torch.ones_like(depth))
    torch.testing.assert_close(out, feature, rtol=0, atol=0)
    model.udr.gre = gre
    assert_raises(lambda: model(rgb, depth[..., :-1]), ValueError)
    with torch.no_grad():
        assert model(rgb[..., :1, :1], depth[..., :1, :1]).shape == (2, 3, 4, 4)
    print('PASS: either gate condition can disable correction; alpha bound, constant edges, tiny/alignment cases')
    return kw, baseline, model


def load_model_class(udr, cpu_reference):
    if not cpu_reference:
        from basicsr.models.udr_mambairv2_model import UDRMambaIRv2Model
        return UDRMambaIRv2Model
    # Run actual model/base/SR source, replacing only unavailable import infrastructure.
    from collections import OrderedDict
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    class Registry:
        def register(self):
            return lambda cls: cls
    registry = Registry()
    common = dict(deepcopy=copy.deepcopy, get_root_logger=lambda: logging.getLogger('udr_check'),
                  MODEL_REGISTRY=registry, OrderedDict=OrderedDict,
                  master_only=lambda fn: fn, DataParallel=DataParallel,
                  DistributedDataParallel=DistributedDataParallel)
    sched = load_isolated((ROOT/'basicsr/models/lr_scheduler.py').read_text(encoding='utf-8'),
                          dict(Counter=__import__('collections').Counter, _LRScheduler=torch.optim.lr_scheduler._LRScheduler))
    common['lr_scheduler'] = sched
    base_model = load_isolated((ROOT/'basicsr/models/base_model.py').read_text(encoding='utf-8'), common)
    common['BaseModel'] = base_model.BaseModel
    def build_network(options):
        options = copy.deepcopy(options)
        options.pop('type')
        return udr.UDRMambaIRv2(**options)
    common.update(build_network=build_network, build_loss=lambda opt: nn.L1Loss())
    sr = load_isolated((ROOT/'basicsr/models/sr_model.py').read_text(encoding='utf-8'), common)
    common['SRModel'] = sr.SRModel
    return load_isolated((ROOT/'basicsr/models/udr_mambairv2_model.py').read_text(encoding='utf-8'), common).UDRMambaIRv2Model


def training_checks(Model, kw, baseline, device):
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        rgb_path = directory/'rgb.pth'
        full_path = directory/'udr.pth'
        torch.save(dict(params=baseline.state_dict()), rgb_path)
        cfg = dict(is_train=True, dist=False, num_gpu=0 if device == 'cpu' else 1,
                   network_g=dict(type='UDRMambaIRv2', **kw),
                   path=dict(pretrain_network_g=str(rgb_path), pretrain_kind='rgb', strict_load_g=True),
                   train=dict(phase='A', optim_g=dict(type='Adam', lr=1e-4),
                              scheduler=dict(type='MultiStepLR', milestones=[], gamma=1.),
                              pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean')))
        a = Model(copy.deepcopy(cfg))
        net = a.get_bare_model(a.net_g)
        before = {k: p.detach().clone() for k, p in net.named_parameters()}
        batch = dict(lq=torch.rand(1,3,4,4,device=device), depth=torch.rand(1,1,4,4,device=device),
                     gt=torch.rand(1,3,16,16,device=device))
        a.feed_data(batch)
        a.optimize_parameters(1)
        for name, p in net.named_parameters():
            if name.startswith('udr.'):
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                assert p.grad.abs().sum() > 0, name
            else:
                assert p.grad is None and not p.requires_grad, name
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        assert not net.layers[0].training and net.udr.training
        assert all(not v.requires_grad for v in net.udr_stats.values())
        a.update_learning_rate(100000)
        assert a.optimizer_g.param_groups[0]['lr'] == 1e-4
        torch.save(dict(params=net.state_dict()), full_path)
        bcfg = copy.deepcopy(cfg)
        bcfg['train'].update(phase='B', rgb_lr=1e-5)
        bcfg['path'].update(pretrain_kind='udr', pretrain_network_g=str(full_path))
        b = Model(bcfg)
        b.feed_data(batch)
        b.optimize_parameters(1)
        bnet = b.get_bare_model(b.net_g)
        assert bnet.conv_first.weight.grad.abs().sum() > 0
        assert bnet.udr.projection.weight.grad.abs().sum() > 0
        assert all(p.requires_grad for p in bnet.parameters())
        b.update_learning_rate(100000)
        assert [g['lr'] for g in b.optimizer_g.param_groups] == [1e-4, 1e-5]
        # Same-phase resume uses full UDR state even though the A config says RGB transfer.
        resume_cfg = copy.deepcopy(cfg)
        resume_cfg['path'].update(pretrain_network_g=str(full_path), resume_state='same_phase.state')
        resumed = Model(resume_cfg)
        resumed.resume_training(dict(optimizers=[a.optimizer_g.state_dict()], schedulers=[a.schedulers[0].state_dict()]))
        for k, value in net.state_dict().items():
            torch.testing.assert_close(resumed.get_bare_model(resumed.net_g).state_dict()[k], value)
        a.test()
        assert a.net_g.training and not net.layers[0].training and net.udr.training
    print('PASS: real optimizer steps, all Depth gradients in A, frozen RGB, joint B gradients, constant group LRs, full checkpoint/resume')


def partition_checks(Model):
    class Toy(nn.Module):
        def forward(self, rgb, depth):
            torch.testing.assert_close(rgb[:, :1], depth)
            return F.interpolate(rgb, scale_factor=4, mode='nearest')
    for batch in (1, 2):
        rgb = torch.rand(batch, 3, 203, 407)
        obj = types.SimpleNamespace(lq=rgb, depth=rgb[:, :1], opt={'scale':4}, net_g=Toy(),
                                    get_bare_model=lambda net: getattr(net, 'module', net))
        Model.test(obj)
        torch.testing.assert_close(obj.output, F.interpolate(rgb, scale_factor=4, mode='nearest'), rtol=0, atol=0)
        obj.net_g.eval()
        Model.test(obj)
        assert not obj.net_g.training
    class WrappedToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Toy()
        def forward(self, *args):
            raise AssertionError('Rank-only validation must not call the DDP wrapper')
    obj.net_g = WrappedToy()
    Model.test(obj)
    torch.testing.assert_close(obj.output, F.interpolate(rgb, scale_factor=4, mode='nearest'), rtol=0, atol=0)
    print('PASS: odd-image partition alignment/stitching, B=1/2, train/eval restoration, rank-only wrapper bypass')


def summary_checks():
    from scripts.udr.evaluate_repeated import summarize
    runs = {label: [{ds:dict(psnr=30 + seed * .01 + delta, ssim=.9 + seed * .0001)
                    for ds in ('Set5','Set14','B100','Urban100','Manga109')} for seed in (10,11,12)]
            for label, delta in [('baseline',0),('udr',.06),('grs',-.01),('gtss',-.12)]}
    report, _ = summarize(runs, [10,11,12])
    assert abs(report['average_delta_vs_rgb']['udr'] - .06) < 1e-12
    assert abs(report['summary']['Set5']['baseline']['psnr']['std'] - .01) < 1e-12
    assert report['udr_criteria']['average_target_met']
    assert not report['udr_criteria']['stop_rule_triggered']
    runs['udr'] = copy.deepcopy(runs['baseline'])
    report, _ = summarize(runs, [10,11,12])
    assert report['udr_criteria']['stop_rule_triggered']
    print('PASS: four-model mean/sample-std, paired deltas and stop rule (synthetic fixtures ONLY)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-reference', action='store_true')
    p.add_argument('--check-data', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(10)
    configs = check_configs()
    if args.cpu_reference:
        base = load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text(encoding='utf-8'))
        udr = load_isolated((ROOT/'basicsr/archs/udr_mambairv2_arch.py').read_text(encoding='utf-8'), dict(MambaIRv2=base.MambaIRv2))
        device = 'cpu'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: --cpu-reference is a separate local contract test.')
        from basicsr.archs import mambairv2_arch as base, udr_mambairv2_arch as udr
        device = 'cuda'
    kw, baseline, model = architecture_checks(base, udr, device)
    Model = load_model_class(udr, args.cpu_reference)
    training_checks(Model, kw, baseline, device)
    partition_checks(Model)
    check_dataset_locally()
    summary_checks()
    # Instantiate exact production dimensions to validate the full state-dict schema.
    options = copy.deepcopy(configs[0]['network_g'])
    options.pop('type')
    full = udr.UDRMambaIRv2(**options)
    print('Production parameter counts:', json.dumps(dict(
        total=sum(p.numel() for p in full.parameters()),
        depth=sum(p.numel() for n,p in full.named_parameters() if n.startswith('udr.')))))
    if args.check_data:
        checkpoint = Path(configs[0]['path']['pretrain_network_g'])
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Confirmed RGB baseline checkpoint missing: {checkpoint}')
        state = torch.load(checkpoint, map_location='cpu')['params']
        full.load_rgb_state_dict({(k[7:] if k.startswith('module.') else k): v for k,v in state.items()})
        print('PASS production RGB checkpoint:', checkpoint)
        from basicsr.data import build_dataset
        for cfg in (configs[0], configs[2]):
            for key, options in cfg['datasets'].items():
                options = copy.deepcopy(options)
                options.update(phase=key.split('_')[0], scale=4)
                dataset = build_dataset(options)
                for index in sorted({0, len(dataset)//2, len(dataset)-1}):
                    dataset[index]
                print('PASS server dataset:', options['name'], len(dataset))
    print('ALL UDR CHECKS PASSED; mode=' + ('CPU reference (not CUDA kernel)' if args.cpu_reference else 'real CUDA'))


if __name__ == '__main__':
    main()
