from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from runner import Runner
import util
import options

from ipdb import set_trace as debug

print(util.yellow("======================================================="))
print(util.yellow("     Lie group Kinetic Langevin Dynamics"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    run = Runner(opt)

    # ====== Training functions ======
    if opt.eval:
        run.evaluate(opt, 0)
    elif opt.status=='train':
        if opt.loss=='ssm':
            run.ssm_train(opt)
        elif opt.loss=='dsm':
            run.dsm_train(opt)
        else:
            raise RuntimeError()
    else:
        raise RuntimeError()
if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
