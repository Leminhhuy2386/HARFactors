"""Model ids used by run_whar.py. Each builder takes the WHAR build_model keywords
(input_channels, window_length, num_classes); CONFIGS records the rest so it ends up in every
result file."""

from __future__ import annotations

from src.models.baselines import GRUClassifier, MambaClassifier, S4DClassifier
from src.models.dualmem import DualMemNet
from src.models.hysteresis import HysteresisNet
from src.models.model import EdgeSSMClassifier
from src.models.quantile import QuantileNet
from src.models.round6 import Round6Net
from src.models.round7 import Round7Net
from src.models.round8 import Round8Net
from src.models.round9 import ForestNet, ParliamentNet
from src.models.round10 import GrammarNet, Round10Net
from src.models.round11 import StyleAlignNet
from src.models.round12 import DendriticNet, IsochronNet
from src.models.round14 import LevelNet
from src.models.tropical import TropicalNet
from src.models.vortex import VortexNet

CONFIGS: dict[str, tuple[type, dict]] = {
    # Edge-SSM v2 and its smaller tier
    "edge_ssm": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64}),
    "edge_ssm_s": (EdgeSSMClassifier, {"hidden": 16, "kan_hidden": 32}),
    # ablations
    "edge_ssm_nophi": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64, "use_phi": False}),
    "edge_ssm_tau_input": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64, "tau_sharing": "input"}),
    "edge_ssm_tau_fixed": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64, "tau_sharing": "fixed"}),
    "edge_ssm_ema": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64, "input_gain": "ema"}),
    # exploratory, proposed after the UTD-MHAD results: complex pole per edge (band-pass memory)
    "edge_ssm_osc": (EdgeSSMClassifier, {"hidden": 64, "kan_hidden": 64, "oscillatory": True}),
    # sequence baselines
    "gru": (GRUClassifier, {"hidden": 64, "num_layers": 2}),
    "s4d": (S4DClassifier, {"d_model": 64, "d_state": 32, "num_layers": 2}),
    "mamba": (MambaClassifier, {"d_model": 32, "d_state": 16, "num_layers": 2}),
    # new architecture families (see NEW_ARCHITECTURES.md)
    "hystnet": (HysteresisNet, {"d_model": 32, "num_layers": 3}),
    # controls: same network, widths frozen at init / widths 0 (play = identity, no memory)
    "hystnet_fixed_r": (HysteresisNet, {"d_model": 32, "num_layers": 3, "learn_widths": False}),
    "hystnet_nomem": (HysteresisNet, {"d_model": 32, "num_layers": 3, "memoryless": True}),
    # (a) variants within the family, chosen between on validation macro-F1 only
    "hystnet_sel": (HysteresisNet, {"d_model": 32, "num_layers": 3, "selective": True}),
    "hystnet_clock": (HysteresisNet, {"d_model": 32, "num_layers": 3, "clock": True}),
    "hystnet_mm": (HysteresisNet, {"d_model": 32, "num_layers": 3, "pooling": "meanmax"}),
    "hystnet_k6": (HysteresisNet, {"d_model": 32, "num_layers": 3,
                                   "widths": (0.03, 0.1, 0.25, 0.5, 1.0, 2.0)}),
    # combinations of the two best single changes: protocol A ranks by validation F1 (selective,
    # k6); protocol B ranks by test F1 of splits 0-3 and is reported on splits 4-7 only (clock, selective)
    "hystnet_sel_k6": (HysteresisNet, {"d_model": 32, "num_layers": 3, "selective": True,
                                       "widths": (0.03, 0.1, 0.25, 0.5, 1.0, 2.0)}),
    "hystnet_sel_clock": (HysteresisNet, {"d_model": 32, "num_layers": 3, "selective": True, "clock": True}),
    "vortexnet": (VortexNet, {"d_model": 32, "n_units": 32, "num_layers": 3}),
    # (b) second round of new families
    "tropicalnet": (TropicalNet, {"d_model": 32, "num_layers": 3}),
    "quantilenet": (QuantileNet, {"d_model": 32, "num_layers": 3}),
    # round 3 (upgrade): rate-independent quantile trackers; both new memories in one block
    "pathquantilenet": (QuantileNet, {"d_model": 32, "num_layers": 3, "path": True,
                                      "trackers": ((0.1, 0.25), (0.5, 0.1), (0.5, 0.5), (0.9, 0.25))}),
    "dualmem": (DualMemNet, {"d_model": 30, "num_layers": 3}),
    "dualmem_path": (DualMemNet, {"d_model": 30, "num_layers": 3, "path": True,
                                  "trackers": ((0.1, 0.25), (0.5, 0.25), (0.9, 0.25))}),
    # round 6: memories from comparisons and counting (edge_ssm/round6.py); configs fixed before any result
    "ordinalnet": (Round6Net, {"memory": "ordinal", "d_model": 32, "num_layers": 3}),
    "recurrencenet": (Round6Net, {"memory": "recurrence", "d_model": 32, "num_layers": 3}),
    "rainflownet": (Round6Net, {"memory": "rainflow", "d_model": 32, "num_layers": 3}),
    # round 7: learned Poincaré sections and return maps (edge_ssm/round7.py); config fixed before any result
    "returnmapnet": (Round7Net, {"memory": "returnmap", "d_model": 32, "num_layers": 3}),
    "phaselocknet": (Round7Net, {"memory": "phaselock", "d_model": 32, "num_layers": 3}),
    # round 8: laws instead of memories (edge_ssm/round8.py); configs fixed before any result.
    # LawNet = memoryless skeleton + per-sensor operator / coupling invariants; controls: random
    # "sensors", plain per-channel statistics; hybrids with RecurrenceNet's blocks. The memoryless
    # ones use d_model 48 so they have about the size of the no-memory control (~19k on UTD-MHAD)
    "lawnet": (Round8Net, {"side": "law", "blocks": "plain", "d_model": 48, "num_layers": 3}),
    "lawnet_rand": (Round8Net, {"side": "law_rand", "blocks": "plain", "d_model": 48, "num_layers": 3}),
    "statnet": (Round8Net, {"side": "stats", "blocks": "plain", "d_model": 48, "num_layers": 3}),
    "lawnet_rec": (Round8Net, {"side": "law", "blocks": "recurrence", "d_model": 32, "num_layers": 3}),
    "recstatnet": (Round8Net, {"side": "stats", "blocks": "recurrence", "d_model": 32, "num_layers": 3}),
    # round 9: models that scale with the number of sensors (edge_ssm/round9.py); configs fixed before any result
    "parliamentnet": (ParliamentNet, {"d_model": 48, "num_layers": 2, "patch": 5, "bits": 4}),
    "forestnet": (ForestNet, {"d_model": 32, "num_layers": 3, "trees": 8, "depth": 4}),
    # round 9b (after parliamentnet's screen: +4.9 SAD, +5.8 Opportunity, -16 UTD): the same parliament with a
    # whole-body agent (RecurrenceNet over all channels, width 24) that posts and votes too; ~30k on UTD-MHAD
    "parliamentbody": (ParliamentNet, {"d_model": 32, "num_layers": 2, "patch": 5, "bits": 4, "body_dim": 24}),
    # round 10: bold memories, literature-checked first (edge_ssm/round10.py); configs fixed before any result
    "econet": (Round10Net, {"memory": "eco", "d_model": 32, "num_layers": 3, "heads": 4, "species": 8,
                            "dt": 0.2, "stride": 3}),
    "grammarnet": (GrammarNet, {"d_model": 64, "nonterminals": 16, "patch": 5}),
    "bhnet": (Round10Net, {"memory": "braid", "d_model": 32, "num_layers": 3, "heads": 8, "strands": 4}),
    # round 11: elastic style fitting next to RecurrenceNet's path (edge_ssm/round11.py), designed from the
    # round-11 evaluation; configs fixed before any result. Control: the elastic path alone.
    "stylealignnet": (StyleAlignNet, {"d_model": 32, "num_layers": 3, "embed": 8, "patch": 5, "programs": 2,
                                      "band": 0.3, "gamma": 0.5}),
    "stylealign_elastic": (StyleAlignNet, {"d_model": 32, "num_layers": 3, "embed": 8, "patch": 5, "programs": 2,
                                           "band": 0.3, "gamma": 0.5, "deep": False}),
    # round 11 ablation (RESEARCH_PLAN.md §5i): which window statistic carries recstatnet's gain?
    "recstat_mean": (Round8Net, {"side": "stats", "blocks": "recurrence", "d_model": 32, "num_layers": 3,
                                 "stats": ("mean",)}),
    "recstat_std": (Round8Net, {"side": "stats", "blocks": "recurrence", "d_model": 32, "num_layers": 3,
                                "stats": ("std",)}),
    "recstat_range": (Round8Net, {"side": "stats", "blocks": "recurrence", "d_model": 32, "num_layers": 3,
                                  "stats": ("min", "max")}),
    # round 11b mechanism test (RESEARCH_PLAN.md §5i): RecurrenceNet with one change each
    "recnet_meanstd": (Round6Net, {"memory": "recurrence", "d_model": 32, "num_layers": 3, "pooling": "meanstd"}),
    "recnet_nonorm": (Round6Net, {"memory": "recurrence", "d_model": 32, "num_layers": 3, "layer_norm": False}),
    "recnet_meanstd_nonorm": (Round6Net, {"memory": "recurrence", "d_model": 32, "num_layers": 3,
                                          "pooling": "meanstd", "layer_norm": False}),
    # round 12: families defined by structure, known layers allowed (edge_ssm/round12.py, RESEARCH_PLAN.md §5j);
    # configs fixed before any result. IsochronNet computes in intrinsic time (phase-indexed ring, progress line,
    # relative phases); "isochron" has no sample-time temporal layer at all, "isochron_rec" adds the ring to
    # recstatnet's trunk. The *_uniform controls replace the phases by sample time (same parameters).
    "isochron": (IsochronNet, {"tonic": "encoder", "d_model": 48, "values": 24, "ring_dim": 12}),
    "isochron_uniform": (IsochronNet, {"tonic": "encoder", "d_model": 48, "values": 24, "ring_dim": 12,
                                         "clock": "uniform"}),
    "isochron_rec": (IsochronNet, {"tonic": "recurrence", "d_model": 32, "num_layers": 3}),
    "isochron_rec_uniform": (IsochronNet, {"tonic": "recurrence", "d_model": 32, "num_layers": 3,
                                             "clock": "uniform"}),
    # dendritic plateau neurons (atlas proposal A): leaky basal branches with a bump nonlinearity, apical gating
    # by window statistics; controls: ReLU branches, no apical gate (statistics at the head instead)
    "dendriticnet": (DendriticNet, {"d_model": 32, "num_layers": 3, "units": 32, "branches": 4}),
    "dendritic_relu": (DendriticNet, {"d_model": 32, "num_layers": 3, "units": 32, "branches": 4, "bump": False}),
    "dendritic_nogate": (DendriticNet, {"d_model": 32, "num_layers": 3, "units": 32, "branches": 4, "gate": False}),
    # round 14: LevelNet, per-channel value-axis maps, channels mixed late (edge_ssm/round14.py, RESEARCH_PLAN.md
    # §5m); configs fixed before any result. Candidates: marginal level maps; + phase-plane density. Controls:
    # channels mixed first, no crossing rates, MLP instead of the level-axis convolution.
    "levelnet": (LevelNet, {}),
    "levelnet_plane": (LevelNet, {"plane": True}),
    "levelnet_early": (LevelNet, {"mixing": "early"}),
    "levelnet_nocross": (LevelNet, {"crossings": False}),
    "levelnet_mlp": (LevelNet, {"tokenizer": "mlp"}),
}


def is_whar_model(model_id: str) -> bool:
    """A WHAR Arena model id (built by the paper-era whar-models, used to check the runner)."""
    if model_id in CONFIGS:
        return False
    from whar_models import WHARModelID

    return model_id in {m.value for m in WHARModelID}


def build(model_id: str, *, input_channels: int, window_length: int, num_classes: int,
          channel_names: list[str] | None = None):
    """channel_names reach only models that ask for them (NEEDS_CHANNEL_NAMES: round 8 finds the
    3-axis sensors by one rule over the names); without them those models take consecutive triples."""
    if is_whar_model(model_id):
        from whar_models import WHARModelID, build_model

        return build_model(WHARModelID(model_id), input_channels=input_channels,
                           window_length=window_length, num_classes=num_classes)
    cls, kwargs = CONFIGS[model_id]
    if getattr(cls, "NEEDS_CHANNEL_NAMES", False):
        kwargs = {**kwargs, "channel_names": channel_names}
    return cls(input_channels, window_length, num_classes, **kwargs)


def describe(model_id: str) -> tuple[str, dict]:
    """(class name, config) recorded in result files and fingerprints."""
    if is_whar_model(model_id):
        return "whar_models." + model_id, {}
    cls, kwargs = CONFIGS[model_id]
    return cls.__name__, kwargs
