# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

###############################################################################
# Activation Functions
###############################################################################

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * x.sigmoid()

class Glu(nn.Module):
    
    def __init__(self, dim):
        super(Glu, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()