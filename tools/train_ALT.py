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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    engine = Engine(config, mode="train")

    with ALT.AcrossLevelTracer(framework = 0,
        host_status = 0,
        ops_status = 1,
        device_status = 0,
        async_prep = 1,
        async_status = 1,
        traced_command=sys.argv,
        no_cuptiTracer=False): 
        
        engine.train()
