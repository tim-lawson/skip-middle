#!/bin/bash

python -m projects.skip_middle.analysis.plot_flops_loss --project fineweb-baseline
python -m projects.skip_middle.analysis.plot_flops_loss --project fineweb-nocontrol
python -m projects.skip_middle.analysis.plot_flops_loss --project fineweb-gated
