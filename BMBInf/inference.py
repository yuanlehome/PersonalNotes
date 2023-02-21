import bminf
import torch
import os

from torch import nn

local_rank = int(os.getenv('LOCAL_RANK', '0'))

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10*1024, 10*1024) for i in range(10)]) # weight + bias = 400 MiB + 40 KiB

    def forward(self, x):
        x = self.linears(x)
        return x

model = MyModule()

shape = (1, 10*1024)
input = torch.ones(shape)

# apply wrapper
with torch.cuda.device(local_rank):
    for i in range(1):
        memory_limit = 401 * 9 * 1024 *1024  #800 MiB+
        model = bminf.wrapper(model = model, quantization = False,  memory_limit = memory_limit) 
        pred = model.forward(input)