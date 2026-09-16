from typing import Optional

import awkward as ak
import numpy as np
import uproot
import vector
from omegaconf import DictConfig, OmegaConf
from particle import pdgid

from ntupelizer.tools import clustering as cl
from ntupelizer.tools import gen_tau_info_matcher as gtim
from ntupelizer.tools import general as g
from ntupelizer.tools import lifetime as lt
from ntupelizer.tools import matching as m
from ntupelizer.tools import particle_filters as pfl


class EDM4HEPNtupelizer:
    def __init__(self, cfg: DictConfig, debug=False):
        print(OmegaConf.to_yaml(cfg, resolve=True))
        self.cfg = cfg
        self.tree_path = self.cfg.tree_path
        self.branches = self.cfg.branches
        self.reco_particles_collection = self.cfg.reco_particles_collection
        self.mc_particles_collection = self.cfg.mc_particles_collection
        self.idx_map_branch = self.cfg.idx_map_branch
        self.include_lifetime_variables = self.cfg.include_lifetime_variables
        self.track_collection = self.cfg.track_collection
        self.vertex_collection = self.cfg.vertex_collection
        self.lifetime_vars = self.cfg.lifetime_vars
        self.signed_lifetime_vars = self.cfg.signed_lifetime_vars
        self.debug = debug

    def _load_input_file(
        self, path: str, tree_path: str = "events", branches: list = None
    ) -> ak.Array:
        raise NotImplementedError("Please implement input loader for your subclass")

    def retrieve_dummy_tau_values(self, gen_jets):
        # `ak.without_parameters` drops vector's Momentum4D record parameter.  The
        # signal p4s lose it anyway on their way through get_matched_gen_tau_property,
        # so keeping it here would make signal and background concatenate into a
        # union of two otherwise identical record types.
        filler = ak.without_parameters(g.reinitialize_p4(ak.zeros_like(gen_jets)))
        gen_jet_tau_info = {
            "gen_jet_tau_vis_energy": ak.zeros_like(gen_jets.eta),
            "gen_jet_tau_decaymode": ak.ones_like(gen_jets.eta) * -1,
            "gen_jet_tau_charge": ak.ones_like(gen_jets.eta) * -999,
            # `filler` is already in the standard schema, so it matches the signal
            # p4s field-for-field.  Re-zipping it under {pt, eta, phi, energy}
            # would rename vector's rho/t coordinates and make the background
            # p4s unmergeable with the signal ones.
            "gen_jet_tau_full_p4": filler,
            "gen_jet_tau_p4": filler,
            # Must match the signal key names produced by fill_tau_info: gen_jet_{tau_DV_x}
            "gen_jet_tau_DV_x": ak.zeros_like(gen_jets.eta),
            "gen_jet_tau_DV_y": ak.zeros_like(gen_jets.eta),
            "gen_jet_tau_DV_z": ak.zeros_like(gen_jets.eta),
        }
        return gen_jet_tau_info

    def get_jet_constituent_property(
        self, reco_property, constituent_idx, num_ptcls_per_jet
    ):
        reco_property_flat = reco_property[ak.flatten(constituent_idx, axis=-1)]
        ret = ak.from_iter(
            [
                ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1)
                for i in range(len(num_ptcls_per_jet))
            ]
        )
        return ret

    def get_candid_from_pdg(self, particles: ak.Array):
        pdg_ids = particles.PDG
        flat_charges = ak.flatten(np.abs(particles.charge) > 0)
        flat_np_pdg = ak.to_numpy(ak.flatten(pdg_ids))
        is_hadron = np.vectorize(pdgid.is_hadron)(flat_np_pdg)
        candid_np = np.where(
            is_hadron,
            np.where(np.abs(flat_charges) > 0, 211, 130),
            np.abs(flat_np_pdg),
        )
        candid = ak.unflatten(candid_np, ak.num(pdg_ids))
        return candid

    def retrieve_jet_tau_info(
        self, arrays: ak.Array, gen_jets: ak.Array, signal_sample: bool = True
    ):
        if signal_sample:
            matcher = gtim.GenTauInfoMatcher(
                arrays=arrays, gen_jets=gen_jets, idx_map_branch=self.idx_map_branch
            )
            jet_tau_info = matcher.fill_tau_info()
        else:
            jet_tau_info = self.retrieve_dummy_tau_values(gen_jets=gen_jets)
        return jet_tau_info

    def ntupelize(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        signal_sample: bool = True,
    ):
        arrays = self._load_input_file(input_path)
        reco_particles, reco_particles_p4 = pfl.RecoParticleFilter(
            arrays=arrays, p_type=self.cfg.reco_particles_collection
        ).results
        mc_particles, mc_particles_p4 = pfl.MCParticleFilter(
            arrays=arrays, p_type=self.cfg.mc_particles_collection
        ).results
        print("Clustering reco jets ...")
        reco_jets, reco_constituent_indices = cl.RecoJetClusterer(
            particles=reco_particles, particles_p4=reco_particles_p4
        ).results
        print("Clustering gen jets ...")
        gen_jets, gen_constituent_indices = cl.GenJetClusterer(
            particles=mc_particles, particles_p4=mc_particles_p4
        ).results
        reco_jets, gen_jets, reco_jet_constituent_indices = m.JetMatcher(
            reco_jets=reco_jets,
            gen_jets=gen_jets,
            reco_jet_constituent_indices=reco_constituent_indices,
            delta_r=0.4,
        ).results
        num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)
        gen_jet_tau_info = self.retrieve_jet_tau_info(
            arrays=arrays, gen_jets=gen_jets, signal_sample=signal_sample
        )

        # =====================================================================
        # Calculate lifetime variables ONCE for all filtered particles
        # (find_all_track_pcas internally filters PDG != 0)
        # =====================================================================
        valid_particle_mask = reco_particles["PDG"] != 0
        all_particle_lifetime_info = lt.find_all_track_pcas(
            events=arrays,
            reco_particle_collection=self.reco_particles_collection,
            track_collection=self.track_collection,
            vertex_collection=self.vertex_collection,
            valid_particle_mask=valid_particle_mask,
            debug=self.debug,
        )

        # =====================================================================
        # Assign lifetime variables to jet constituents using indices
        # =====================================================================
        lifetime_info = lt.assign_lifetime_vars_to_jets(
            all_particle_lifetime_info=all_particle_lifetime_info,
            reco_jet_constituent_indices=reco_jet_constituent_indices,
            reco_jets=reco_jets,
            lifetime_vars=self.lifetime_vars,
            signed_lifetime_vars=self.signed_lifetime_vars,
        )

        # =====================================================================
        # Add event-level reco candidate info for each jet
        # For each jet, attach ALL reco candidates from that event (not just constituents)
        # This is useful for e.g. attention-based models that need full event context
        # =====================================================================
        event_lifetime_info = lt.replicate_lifetime_vars_per_jet(
            all_particle_lifetime_info=all_particle_lifetime_info,
            reco_jets=reco_jets,
            lifetime_vars=self.lifetime_vars,
            signed_lifetime_vars=self.signed_lifetime_vars,
        )

        # Add event-level reco candidate p4s for each jet
        event_reco_cand_p4s = ak.from_iter(
            [
                [reco_particles_p4[j] for _ in range(len(reco_jets[j]))]
                for j in range(len(reco_jets))
            ]
        )

        # Get reco candidate p4s per jet (constituents only)
        reco_cand_p4s = self.get_jet_constituent_property(
            reco_property=reco_particles_p4,
            constituent_idx=reco_jet_constituent_indices,
            num_ptcls_per_jet=num_ptcls_per_jet,
        )
        reco_cand_p4s = g.reinitialize_p4(reco_cand_p4s)

        reco_particles_pdgs = self.get_candid_from_pdg(reco_particles)
        event_reco_cand_pdgs = ak.from_iter(
            [
                [reco_particles_pdgs[j] for _ in range(len(reco_jets[j]))]
                for j in range(len(reco_jets))
            ]
        )

        # Get reco candidate p4s per jet (constituents only)
        reco_cand_pdgs = self.get_jet_constituent_property(
            reco_property=reco_particles_pdgs,
            constituent_idx=reco_jet_constituent_indices,
            num_ptcls_per_jet=num_ptcls_per_jet,
        )

        event_reco_cand_charges = ak.from_iter(
            [
                [reco_particles.charge[j] for _ in range(len(reco_jets[j]))]
                for j in range(len(reco_jets))
            ]
        )

        # Get reco candidate p4s per jet (constituents only)
        reco_cand_charges = self.get_jet_constituent_property(
            reco_property=reco_particles.charge,
            constituent_idx=reco_jet_constituent_indices,
            num_ptcls_per_jet=num_ptcls_per_jet,
        )

        # =====================================================================
        # Combine all variables into a single ak.Array (flattened per jet)
        # =====================================================================
        # Standardise jet p4s to {pt, eta, phi, energy} so they are consistent
        # with reco_cand_p4s and gen_jet_tau_p4 (which go through reinitialize_p4).
        # FastJet returns jets in {energy, x, y, z} Cartesian form.
        reco_jets_p4 = g.reinitialize_p4(reco_jets)
        gen_jets_p4 = g.reinitialize_p4(gen_jets)
        combined_dict = {
            # Reco jet p4s
            "reco_jet_p4": reco_jets_p4,
            # Gen jet p4s
            "gen_jet_p4": gen_jets_p4,
            # Reco candidate p4s, pdgs and charges (jet constituents)
            "reco_cand_p4s": reco_cand_p4s,
            "reco_cand_pdgs": reco_cand_pdgs,
            "reco_cand_charges": reco_cand_charges,
            # Gen jet tau info
            **gen_jet_tau_info,
            # Lifetime info per jet constituent (reco_cand_*)
            **{f"reco_cand_{k}": v for k, v in lifetime_info.items()},
            # Event-level lifetime info (event_reco_cand_*)
            **{f"event_reco_cand_{k}": v for k, v in event_lifetime_info.items()},
            # Event-level reco cand p4s, pdgs and charges
            "event_reco_cand_p4s": event_reco_cand_p4s,
            "event_reco_cand_pdgs": event_reco_cand_pdgs,
            "event_reco_cand_charges": event_reco_cand_charges,
        }

        # Flatten all arrays from [n_events, n_jets, ...] to [n_jets_total, ...]
        data = ak.Array({k: ak.flatten(v, axis=1) for k, v in combined_dict.items()})

        lepton_mask = (13 == abs(data.reco_cand_pdgs)) | (
            11 == abs(data.reco_cand_pdgs)
        )
        hadronic_jet_mask = ak.sum(lepton_mask, axis=1) == 0
        data = ak.Array({key: data[key][hadronic_jet_mask] for key in data.fields})
        removal_mask = data.gen_jet_tau_decaymode != 16
        if signal_sample:
            removal_mask = (data.gen_jet_tau_decaymode != -1) & removal_mask
        print(f"{np.sum(removal_mask)} jets after masking")
        # Keep data as ak.Array (not ak.Record) so that ak.from_parquet reads it
        # back as an ak.Array and ak.concatenate works in the batch-merge step.
        # ak.Record is a single-row wrapper; writing it to parquet produces a
        # file that deserialises as a Record, not an Array, breaking concatenation.
        data = ak.Array({key: data[key][removal_mask] for key in data.fields})
        ak.to_parquet(
            data,
            output_path,
            compression="zstd",
            compression_level=6,
        )


class PodioROOTNtuplelizer(EDM4HEPNtupelizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.idx_map_branch = "idx_mc"  # LCIO and PodioROOT could actually both rename this branch to idx_mc

    def _load_input_file(
        self, path: str, tree_path: str = "events", branches: list = None
    ) -> ak.Array:
        with uproot.open(path) as in_file:
            tree = in_file[tree_path]
            print(f"{path} file has {tree.num_entries} entries")
            arrays = tree.arrays(branches)
            # --- Reco ↔ MC truth indices ---
            arrays["idx_reco"] = arrays["_RecoMCTruthLink_from.index"]
            arrays["idx_mc"] = arrays["_RecoMCTruthLink_to.index"]
            arrays["mc_weight"] = arrays["RecoMCTruthLink.weight"]
            # --- Reco ↔ track indices ---
            arrays["idx_track"] = arrays["_PandoraPFOs_tracks.index"]
        return arrays


class LCIOROOTNtuplelizer(EDM4HEPNtupelizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.idx_map_branch = "idx_mc"  # LCIO and PodioROOT could actually both rename this branch to idx_mc

    def _load_input_file(
        self, path: str, tree_path: str = "events", branches: list = None
    ) -> ak.Array:
        with uproot.open(path) as in_file:
            tree = in_file[tree_path]
            print(f"{path} file has {tree.num_entries} entries")
            arrays = tree.arrays(branches)
            # --- Reco ↔ MC truth indices ---
            arrays["idx_reco"] = arrays["_RecoMCTruthLink_from.index"]
            arrays["idx_mc"] = arrays["_RecoMCTruthLink_to.index"]
            arrays["mc_weight"] = arrays["RecoMCTruthLink.weight"]
            # --- Reco ↔ track indices ---
            arrays["idx_track"] = arrays["_PandoraPFOs_tracks.index"]
        return arrays


class DecayProductNtupelizer(PodioROOTNtuplelizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

    def retrieve_dummy_tau_values(self, gen_jets):
        info = super().retrieve_dummy_tau_values(gen_jets)

        # Add TYPED empty jagged arrays for the three extra daughter properties.
        # Keeping the record/scalar type even when there are zero daughters means
        # the downstream dataloader can read daughter_p4["pt"] on empty lists
        # instead of failing on an `unknown`-type array with no fields.
        def _empty_daughters(template):
            one = ak.Array([[[template] for _ in ev] for ev in gen_jets])
            return one[:, :, 0:0]

        info["gen_jet_tau_vis_daughter_p4s"] = _empty_daughters(
            {"pt": 0.0, "eta": 0.0, "phi": 0.0, "energy": 0.0}
        )
        info["gen_jet_tau_vis_daughter_pdgs"] = _empty_daughters(0)
        info["gen_jet_tau_vis_daughter_charges"] = _empty_daughters(0.0)
        return info

    def retrieve_jet_tau_info(
        self, arrays: ak.Array, gen_jets: ak.Array, signal_sample: bool = True
    ):
        if signal_sample:
            matcher = gtim.GenTauInfoMatcherWithDaughters(
                arrays=arrays,
                gen_jets=gen_jets,
                idx_map_branch=self.idx_map_branch,
                replace_intermediate_mesons=self.cfg.get(
                    "replace_intermediate_mesons", False
                ),
            )
            jet_tau_info = matcher.fill_tau_info()
        else:
            jet_tau_info = self.retrieve_dummy_tau_values(gen_jets=gen_jets)
        return jet_tau_info
