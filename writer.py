import os

import wandb
import torch

class BaseWriter(object):
    def __init__(self, opt):
        # self.rank = opt.global_rank
        self.rank = 0 # for now, we only support single GPU training
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def close(self): pass

class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter,self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt.wandb_api_key)
            wandb.init(dir=str(opt.log_dir), project=opt.wandb_project, entity=opt.wandb_user, name=opt.dir, config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            # adopt from torchvision.utils.save_image
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)
    def add_plot(self, step, key, fig):
        if self.rank == 0:
            wandb.log({key: fig}, step=step)

    def add_plot_image(self, step, key, fig):
        if self.rank == 0:
            wandb.log({key: wandb.Image(fig)}, step=step)