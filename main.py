# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from option import args

import torch
import utility
import data
import loss
from trainer import Trainer
import warnings
warnings.filterwarnings('ignore')
import os
os.system('pip install einops')
import model
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        if args.pretrain != "":
            if 's3' in args.pretrain:
                import moxing as mox
                mox.file.copy_parallel(args.pretrain,"/cache/models/ipt.pt")
                args.pretrain = "/cache/models/ipt.pt"
            state_dict = torch.load(args.pretrain)
            _model.model.load_state_dict(state_dict,strict = False)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.test()
        checkpoint.done()
            
if __name__ == '__main__':
    main()
