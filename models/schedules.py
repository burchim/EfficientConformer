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

import math

class constant_learning_rate_scheduler:

    def __init__(self, optimizer, lr_value):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.lr_value = lr_value

    def step(self):
        
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        self.optimizer.param_groups[0]['lr'] = self.lr_value

class constant_with_decay_learning_rate_scheduler:

    def __init__(self, optimizer, lr_values, decay_steps):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.lr_values = lr_values
        self.decay_steps = decay_steps

    def step(self):
        
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        lr_value = self.lr_values[0]
        for i, step in enumerate(self.decay_steps):
            if self.model_step > step:
                lr_value = self.lr_values[i + 1]
            else:
                break
        self.optimizer.param_groups[0]['lr'] = lr_value

class cosine_annealing_learning_rate_scheduler:

    def __init__(self, optimizer, warmup_steps, lr_max, lr_min, end_step):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.end_step = end_step

    def step(self):
        
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Compute LR
        if s <= self.warmup_steps: # Warmup phase
            lr = s / self.warmup_steps * self.lr_max
        else: # Annealing phase
            lr = (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(math.pi * (s - self.warmup_steps) / (self.end_step - self.warmup_steps))) + self.lr_min

        # Update LR
        self.optimizer.param_groups[0]['lr'] = lr

class transformer_learning_rate_scheduler:

    def __init__(self, optimizer, dim_model, warmup_steps, K):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.K = K

    def step(self):
        
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        arg1 = s**-0.5
        arg2 = s * (self.warmup_steps**-1.5)
        self.optimizer.param_groups[0]['lr'] = self.K * self.dim_model**-0.5 * min(arg1, arg2)

class exponential_decay_transformer_learning_rate_scheduler:

    def __init__(self, optimizer, warmup_steps, lr_max, alpha, end_step):

        # Model Optimizer
        self.optimizer = optimizer

        # Model Step
        self.model_step = -1

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.alpha = alpha
        self.end_step = end_step

    def step(self):
        
        # Update Model Step
        self.model_step += 1
        s = self.model_step + 1

        # Update LR
        arg1 = s / self.warmup_steps * self.lr_max # Warmup phase
        arg2 = self.lr_max * self.alpha**((s - self.warmup_steps) / (self.end_step - self.warmup_steps)) # Decay phase
        self.optimizer.param_groups[0]['lr'] = min(arg1, arg2)