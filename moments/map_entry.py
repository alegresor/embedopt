"""Entry point for Li et al.'s map-guided EmbedOpt with optional partial diffusion (PARTIAL_INIT / PARTIAL_SIGMA).

Identical to `python -m embedopt.inference.inference`, except that MapReward records the trajectory's seed (so a
per-seed PARTIAL_INIT template can be resolved) and the partial-diffusion patch is installed. Without PARTIAL_INIT
it behaves exactly like their entry point.
"""

import re

import embedopt.reward.map as _map
from moments.partial import install_partial_diffusion
from moments.reward import CURRENT

_MapReward = _map.MapReward


class SeededMapReward(_MapReward):
    def __init__(self, *args, output_dir=None, **kwargs):
        seed = re.search(r"seed_(\d+)", output_dir or "")
        CURRENT["seed"] = int(seed.group(1)) if seed else None
        super().__init__(*args, output_dir=output_dir, **kwargs)


if __name__ == "__main__":
    _map.MapReward = SeededMapReward  # inference.py builds MapReward for reward type "map"
    install_partial_diffusion()  # no-op unless PARTIAL_INIT is set
    from embedopt.inference.inference import main

    main()
