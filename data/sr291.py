# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from data import srdata

class SR291(srdata.SRData):
    def __init__(self, args, name='SR291', train=True, benchmark=False):
        super(SR291, self).__init__(args, name=name)

