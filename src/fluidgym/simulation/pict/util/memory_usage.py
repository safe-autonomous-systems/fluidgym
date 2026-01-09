# Copyright 2025 Aleksandra Franz, Nils Thürey
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

import os
import psutil
import torch


class MemoryUsage:
    def __init__(self, logger=None):
        self.max_mem_GPU = 0
        self.max_mem_GPU_name = ""
        self.max_mem_CPU = 0
        self.max_mem_CPU_name = ""
        self.process = psutil.Process(os.getpid())
        self.LOG = logger

    def fmt_mem(self, value_bytes):
        return "%.02fMiB" % (value_bytes / (1024 * 1024),)

    def check_memory(self, name="", verbose=True):
        used_mem_GPU = torch.cuda.memory_allocated()
        if used_mem_GPU > self.max_mem_GPU:
            self.max_mem_GPU = used_mem_GPU
            self.max_mem_GPU_name = name
        used_mem_CPU = self.process.memory_info()[0]
        if used_mem_CPU > self.max_mem_CPU:
            self.max_mem_CPU = used_mem_CPU
            self.max_mem_CPU_name = name
        if verbose and (self.LOG is not None):
            self.LOG.info(
                "used memory '%s': CPU %s, GPU %s",
                name,
                self.fmt_mem(used_mem_CPU),
                self.fmt_mem(used_mem_GPU),
            )

    def print_max_memory(self):
        if self.LOG is not None:
            self.LOG.info(
                "Max used memory: CPU %s at %s, GPU %s at '%s'",
                self.fmt_mem(self.max_mem_CPU),
                self.max_mem_CPU_name,
                self.fmt_mem(self.max_mem_GPU),
                self.max_mem_GPU_name,
            )
