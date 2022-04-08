# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# Add ALT path to sys.path
ALT_PATH="/paddle/chao/experiments-1/AcrossLevelTracer"
sys.path.append(ALT_PATH)
import AcrossLevelTracer as ALT

# Add CUptiTracer path
# New
CUpti_PATH = ALT_PATH + "/profilers/CUptiTracer/lib"
sys.path.append(CUpti_PATH)
import cuptiTracer

# Old
#CUpti_PATH = "/paddle/chao/experiments-1/CUptiTracer/lib"
#sys.path.append(CUpti_PATH)
#from cuptiTracer import cuptiTracer as ct

# Original code
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    #with ct():
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    
    # enable CUptiTarcer
    ct = cuptiTracer.cuptiTracer("/etc/libCUptiTracer.conf")
    ct.cuptiTracerStart()
    engine = Engine(config, mode="train")
    engine.train()
    ct.cuptiTracerStop()
