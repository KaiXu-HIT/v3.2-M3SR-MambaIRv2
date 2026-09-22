"""UDR v3.2: strict RGB transfer, two 100k phases, differential LR, L1 only."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class UDRMambaIRv2Model(SRModel):
    def __init__(self, opt):
        if opt['is_train']:
            train = opt['train']
            if train.get('phase') not in ('A', 'B'):
                raise ValueError('Set train.phase to A or B.')
            if train.get('pixel_opt') != dict(type='L1Loss', loss_weight=1.0, reduction='mean'):
                raise ValueError('UDR uses baseline mean L1 only, with weight 1.')
            if train.get('perceptual_opt') or train.get('ema_decay', 0):
                raise ValueError('The UDR experiment uses no perceptual loss or EMA.')
            if not opt['path'].get('pretrain_network_g'):
                raise ValueError('UDR must load RGB baseline (A) or full Phase A weights (B).')
        super().__init__(opt)

    def model_to_device(self, net):
        # UDR: freeze BEFORE DDP constructs its gradient reducer.
        if self.is_train:
            net.configure_phase(self.opt['train']['phase'])
        if not self.opt['dist'] and self.opt['num_gpu'] > 1:
            raise ValueError('Use torchrun/DDP for multiple GPUs, not DataParallel.')
        return super().model_to_device(net)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        if not strict:
            raise ValueError('UDR requires strict checkpoint validation.')
        # check_resume changes the path to the same-phase full UDR checkpoint.
        kind = 'udr' if self.opt['path'].get('resume_state') else self.opt['path'].get('pretrain_kind', 'udr')
        if kind == 'udr':
            return super().load_network(net, load_path, strict=True, param_key=param_key)
        if kind != 'rgb' or not self.is_train or self.opt['train']['phase'] != 'A':
            raise ValueError('RGB-only transfer is allowed only when starting Phase A.')
        state = torch.load(load_path, map_location='cpu')
        if param_key is not None:
            if param_key not in state:
                raise KeyError(f'RGB checkpoint has no requested key {param_key!r}.')
            state = state[param_key]
        state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
        self.get_bare_model(net).load_rgb_state_dict(state)
        get_root_logger().info('UDR: strictly loaded ALL RGB baseline tensors; only udr.* is newly initialized.')

    def setup_optimizers(self):
        train = self.opt['train']
        settings = deepcopy(train['optim_g'])
        optim_type = settings.pop('type')
        depth_lr = settings.pop('lr')
        rgb_lr = train.get('rgb_lr', depth_lr * 0.1)
        rgb, depth = [], []
        for name, parameter in self.get_bare_model(self.net_g).named_parameters():
            if parameter.requires_grad:
                (depth if name.startswith('udr.') else rgb).append(parameter)
        groups = [dict(params=depth, lr=depth_lr, group_name='depth')]
        if train['phase'] == 'B':
            groups.append(dict(params=rgb, lr=rgb_lr, group_name='rgb'))
        elif rgb:
            raise RuntimeError('Phase A unexpectedly contains trainable RGB parameters.')
        self.optimizer_g = self.get_optimizer(optim_type, groups, lr=depth_lr, **settings)
        self.optimizers.append(self.optimizer_g)
        get_root_logger().info(f'UDR Phase {train["phase"]}: Depth LR={depth_lr}, RGB LR={rgb_lr if rgb else 0}')

    def feed_data(self, data):
        super().feed_data(data)
        self.depth = data['depth'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq, self.depth)
        loss = self.cri_pix(self.output, self.gt)
        loss.backward()
        self.optimizer_g.step()
        stats = self.get_bare_model(self.net_g).udr_stats
        self.log_dict = self.reduce_loss_dict(dict(l_pix=loss, **stats))

    def test_selfensemble(self):
        raise NotImplementedError('Use the original baseline partition protocol for UDR comparisons.')

    # UDR: reuse the original matched RGB/depth partition protocol.
    def test(self):
        batch, C, h, w = self.lq.size()
        was_training = self.net_g.training
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # UDR: apply exactly the RGB partition padding to depth.
        depth_img = F.pad(self.depth, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        depth_chops = []
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
            depth_chops.append(depth_img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop, depth_chop in zip(img_chops, depth_chops):
                    out = self.net_g_ema(chop, depth_chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(batch, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            # UDR: validation is rank-0-only; bypass DDP buffer broadcasts here.
            # Training still uses the wrapped net_g and its gradient reducer.
            test_net = self.get_bare_model(self.net_g)
            with torch.no_grad():
                outputs = []
                for chop, depth_chop in zip(img_chops, depth_chops):
                    out = test_net(chop, depth_chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(batch, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train(was_training)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
