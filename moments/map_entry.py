"""Entry point for Li et al.'s map-guided EmbedOpt with optional partial diffusion (PARTIAL_INIT / PARTIAL_SIGMA).

Identical to `python -m embedopt.inference.inference`, except that
  - MapReward records the trajectory's seed (so a per-seed PARTIAL_INIT template can be resolved) and the
    partial-diffusion patch is installed (no-op without PARTIAL_INIT);
  - the CA correspondence used for MapReward's Kabsch frame is checked. MapReward matches the predicted and reference
    CA atoms by a Smith-Waterman sequence alignment (embedopt.util.structure.get_common_ca_ind), which has a known
    problem when the two sequences are identical, e.g. when the reference is itself a Protenix model docked into the
    map. If the reference has exactly the predicted CA residue sequence, the identity correspondence is used instead,
    and the startup log reports how many CA pairs the inherited alignment had and whether they were in register.
"""

import re

import numpy as np

import embedopt.reward.map as _map
from moments.partial import install_partial_diffusion
from moments.reward import CURRENT

import functools, os
from embedopt.util import structure as _structure
_cycles = os.environ.get("MAP_KABSCH_CYCLES")
if _cycles is not None:
    _map.iterative_kabsch_alignment_common_ca = functools.partial(_structure.iterative_kabsch_alignment_common_ca,
                                                                  cycles=int(_cycles))
                                                                  
_MapReward = _map.MapReward


class SeededMapReward(_MapReward):
    def __init__(self, *args, output_dir=None, **kwargs):
        seed = re.search(r"seed_(\d+)", output_dir or "")
        CURRENT["seed"] = int(seed.group(1)) if seed else None
        super().__init__(*args, output_dir=output_dir, **kwargs)
        names = ("ref_pdb_path", "pred_atom_array")
        ref_path, pred = (kwargs[n] if n in kwargs else args[i] for i, n in enumerate(names))
        self._check_ca_mapping(ref_path, pred)

    def _check_ca_mapping(self, ref_path, pred):
        import SFC_Torch as sfc

        ref = sfc.PDBParser(ref_path)
        ca_ref = [i for i, n in enumerate(ref.cra_name) if n.split("-")[-1] == "CA"]
        ca_pred = np.flatnonzero(np.asarray(pred.atom_name) == "CA")
        pos_r = {a: k for k, a in enumerate(ca_ref)}
        pos_p = {a: k for k, a in enumerate(ca_pred)}
        old = [(pos_p.get(int(i)), pos_r.get(int(j))) for i, j in zip(self.common_ca_ind_1, self.common_ca_ind_2)]
        offsets = sorted({p - r for p, r in old if p is not None and r is not None})
        same_seq = [ref.cra_name[i].split("-")[2] for i in ca_ref] == list(np.asarray(pred.res_name)[ca_pred])
        msg = (f"[MapReward CA mapping] inherited: {len(old)} pairs, residue offsets (pred - ref) {offsets[:5]}"
               f"{' ...' if len(offsets) > 5 else ''}; reference has the predicted sequence: {same_seq}")
        if same_seq:
            self.common_ca_ind_1, self.common_ca_ind_2 = [int(i) for i in ca_pred], ca_ref
            self.ref_ca_numpy = np.asarray(ref.atom_pos)[ca_ref].copy()
            msg += f" -> using the identity correspondence ({len(ca_ref)} CA)"
        print(msg, flush=True)


if __name__ == "__main__":
    _map.MapReward = SeededMapReward  # inference.py builds MapReward for reward type "map"
    install_partial_diffusion()  # no-op unless PARTIAL_INIT is set
    from embedopt.inference.inference import main

    main()