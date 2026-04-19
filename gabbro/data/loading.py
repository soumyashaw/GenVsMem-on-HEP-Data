from pathlib import Path

import awkward as ak
import fastjet as fj
import h5py
import numpy as np
import uproot
import vector
import os
from tqdm import tqdm

from gabbro.utils.arrays import (
    ak_add_zero_padded_features,
    ak_select_and_preprocess,
    combine_ak_arrays,
    np_to_ak,
    shuffle_ak_arr_along_axis1,
)

from gabbro.utils.jet_types import get_jet_type_from_file_prefix, jet_types_dict
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import print_field_structure_of_ak_array

logger = get_pylogger(__name__)

vector.register_awkward()


def read_cms_open_data_file(
    filepath: str,
    particle_features: list = None,
    jet_features: list = None,
    return_p4: bool = False,
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
):
    """Reads a single file from the CMS Open Data dataset.

    Parameters:
    -----------
    filepath : str
        Path to the h5 file.
    particle_features : list of str, optional
        List of particle-level features to load.
    jet_features : list of str, optional
        List of jet-level features to load.
    return_p4 : bool, optional
        Whether to return the 4-momentum of the particles.
    labels : list of str, optional
        List of truth labels to load. This is just there for compatibility with the
        JetClass dataset. We treat the CMS Open Data dataset as QCD jets.

    Returns:
    --------
    ak.Array
        An awkward array of the particle-level features or jet features if only one
        of the two is requested.
        If both are requested, a tuple of two awkward arrays is returned, the first
        one containing the particle-level features and the second one the jet-level.
    """
    # Selected all AK8 jets with pt > 300, abs(eta) < 2.5, passing standard jet ID criteria
    # Events are stored in h5 format with 4 keys
    # 'event_info' : [Run Number, LumiBlock, Event Number]
    # 'jet_kinematics' : [ pt, eta, phi, softdrop mass] of the AK8 jet
    # 'PFCands' : Zero padded list of up to 150 PFcandidates inside the AK8 jet. For each entry, shape is [150, 10].
    # Info for each candidate is [px, py, pz, E, d0, d0Err, dz, dzErr, charge, pdgId, PUPPI weight]
    # 'jet_tagging': Tagging info/scores for the AK8 jet, 13 entries
    # [nConstituents, tau1, tau2, tau3, tau4, PNet H4q vs QCD, PNet Hbb vs QCD, PNet Hcc vs QCD, PNet QCD score, PNet T vs QCD, PNet W vs QCD, PNet Z vs QCD, PNet regressed mass]
    if particle_features is None and jet_features is None:
        raise ValueError("Either particle_features or jet_features must be provided.")

    with h5py.File(filepath, "r") as f1:
        # Convert to numpy
        PFCands = f1["PFCands"][:]
        # event_info = f1["event_info"][:]
        jet_kinematics = f1["jet_kinematics"][:]
        # jet_tagging = f1["jet_tagging"][:]

    p4s_ak = np_to_ak(
        x=PFCands[:, :, :4],
        names=["px", "py", "pz", "E"],
        mask=PFCands[:, :, 3] != 0,
    )

    # Create an awkward vector from the PFCands
    p4 = ak.zip(
        {"px": p4s_ak.px, "py": p4s_ak.py, "pz": p4s_ak.pz, "E": p4s_ak.E},
        with_name="Momentum4D",
    )

    p4_jet = ak.sum(p4, axis=1)

    x_ak_particles = ak.Array(
        {
            "part_pt": p4.pt,
            "part_eta": p4.eta,
            "part_phi": p4.phi,
            "part_etarel": p4.deltaeta(p4_jet),
            "part_phirel": p4.deltaphi(p4_jet),
        }
    )

    x_ak_jets = ak.Array(
        {
            "jet_pt": jet_kinematics[:, 0],
            "jet_eta": jet_kinematics[:, 1],
            "jet_phi": jet_kinematics[:, 2],
            "jet_sdmass": jet_kinematics[:, 3],
            "jet_mass": p4_jet.mass,
        }
    )

    if particle_features is None:
        x_ak_particles = None
    else:
        x_ak_particles = x_ak_particles[particle_features]
    if jet_features is None:
        x_ak_jets = None
    else:
        x_ak_jets = x_ak_jets[jet_features]

    len_p4_jet = len(p4_jet)

    ak_labels = ak.Array(
        {
            "label_QCD": np.ones(len_p4_jet),
            "label_Hbb": np.zeros(len_p4_jet),
            "label_Hcc": np.zeros(len_p4_jet),
            "label_Hgg": np.zeros(len_p4_jet),
            "label_H4q": np.zeros(len_p4_jet),
            "label_Hqql": np.zeros(len_p4_jet),
            "label_Zqq": np.zeros(len_p4_jet),
            "label_Wqq": np.zeros(len_p4_jet),
            "label_Tbqq": np.zeros(len_p4_jet),
            "label_Tbl": np.zeros(len_p4_jet),
        }
    )

    if return_p4:
        return x_ak_particles, x_ak_jets, p4, ak_labels[labels]

    return x_ak_particles, x_ak_jets, ak_labels[labels]


def read_tokenized_jetclass_file(
    filepath,
    load_token_ids: bool = True,
    particle_features: list = None,
    particle_features_tokenized: list = None,
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
    remove_start_token=False,
    remove_end_token=False,
    shift_tokens_minus_one=False,
    add_padded_particle_features_start=False,
    add_padded_particle_features_end=False,
    n_load=None,
    random_seed=None,
    jet_features=None,
    class_token_dict=None,
):
    """Reads a file that contains the tokenized JetClass jets.

    Parameters
    ----------
    filepath : str
        Path to the file.
    load_token_ids : bool, optional
        Whether to load the token-ids.
    particle_features : List[str], optional
        A list of particle-level features to be loaded. These are the full-resolution
        ones.
    particle_features_tokenized : List[str], optional
        A list of particle-level features to be loaded. These are the tokenized ones.
    labels : List[str], optional
        A list of truth labels to be loaded.
    remove_start_token : bool, optional
        Whether to remove the start token from the tokenized sequence.
    remove_end_token : bool, optional
        Whether to remove the end token from the tokenized sequence.
    shift_tokens_minus_one : bool, optional
        Whether to shift the token values by -1.
    add_padded_particle_features_start : bool, optional
        Whether to add zero-padded padded particle features at the start of the sequence.
        Default is False.
    add_padded_particle_features_end : bool, optional
        Whether to add zero-padded padded particle features at the end of the sequence.
        Default is False.
    n_load : int, optional
        Number of events to load. If None, all events are loaded.
    random_seed : int, optional
        Random seed for shuffling the data. If None, no shuffling is performed.
    jet_features : List[str], optional
        A list of jet-level features to be loaded.
        Possible options are:
        - jet_pt
        - jet_eta
        - jet_phi
        - jet_energy
        - jet_nparticles
        - jet_sdmass
        - jet_tau1
        - jet_tau2
        - jet_tau3
        - jet_tau4

    class_token_dict: Dict, optional
        Dictionary specifying which jet types have which tokens.
        Example: {"QCD": 1, "Hbb": 2}.
        Note that the labels must be integers between 1 and n_jettypes_used.
        Possible jet types:
        - "QCD"
        - "Hbb"
        - "Hcc"
        - "Hgg"
        - "H4q"
        - "Hqql"
        - "Zqq"
        - "Wqq"
        - "Tbqq"
        - "Tbl"
    Returns
    -------
    x_ak
        An awkward array of the particle-level features. These are the merged
        arrays of the full-resolution and tokenized features (and particle tokens).
    y : ak.Array
        An awkward array of the truth labels (one-hot encoded).
    x_jets : ak.Array
        An awkward array of the jet-level features.
    """

    # check if it's the old or new file format:
    # old: tokenized files are a awkward.highlevel.Array objects that has the token-ids
    #      without any keys or so --> array.fields is empty
    # new: tokenized files are a awkward.highlevel.Array objects that has the token-ids
    #      as a key "part_token_id"

    # if type of `particle_features` is dict, convert it to a list with the keys
    if particle_features is not None and particle_features is not isinstance(
        particle_features, list
    ):
        particle_features = [feat for feat in particle_features]
    if particle_features_tokenized is not None and not isinstance(
        particle_features_tokenized, list
    ):
        particle_features_tokenized = [feat for feat in particle_features_tokenized]

    tokenized_file = ak.from_parquet(filepath)
    if jet_features is not None:
        try:
            ak_jets = tokenized_file["jet_features"]
        except AttributeError:
            print("No jet features found in the file!")
            print(
                "Please make sure that the file contains jet features or run without setting feature_dict_jet."
            )
    x_jets = ak_jets[jet_features] if jet_features is not None else None
    # check if the file is in the new format, and if so, extract the token-ids and
    # overwrite ak_tokens ak array
    logger.info("The loaded file has the following structure:")
    print_field_structure_of_ak_array(tokenized_file)
    if len(tokenized_file.fields) > 0:
        ak_tokens = tokenized_file["part_token_id"]
    else:
        ak_tokens = tokenized_file
        logger.warning(
            f"File {filepath} is in the old format. At the moment this is still supported, "
            "but might lead to problems in the future."
        )

    if n_load is not None:
        ak_tokens = ak_tokens[:n_load]
        x_jets = x_jets[:n_load] if jet_features is not None else None

    # extract jet type from filename and create the corresponding labels
    jet_type_prefix = filepath.split("/")[-1].split("_")[0] + "_"
    jet_type_name = get_jet_type_from_file_prefix(jet_type_prefix)

    if class_token_dict is not None:
        if jet_type_name not in class_token_dict.keys():
            raise ValueError(
                f"Loaded file does not correspond to any of the jet types in class_token_dict. "
                f"Jet type name is {jet_type_name}, filepath is {filepath}. Class token dict is "
                f"{class_token_dict.keys()}. Specify which files to load in configs/data/iter_dataset_X.yaml."
            )
        class_token = class_token_dict[jet_type_name]
        class_token_array = (np.ones(len(ak_tokens)) * class_token).reshape((-1, 1))
        number_of_jettypes = len(class_token_dict)

    # one-hot encode the jet type
    labels_onehot = ak.Array(
        {
            f"label_{jet_type}": np.ones(len(ak_tokens)) * (jet_type_name == jet_type)
            for jet_type in jet_types_dict
        }
    )

    if class_token_dict is not None:
        if remove_start_token or remove_end_token or shift_tokens_minus_one:
            raise ValueError(
                "Cannot remove start or end token or shift tokens when class token is used."
            )
    if len(ak_tokens.fields) != 0:
        # group tokenization (new setup)
        stop_token_value_dict = {  # noqa: F841
            field: ak_tokens[field][0, -1] for field in ak_tokens.fields
        }
    if remove_start_token:
        ak_tokens = ak_tokens[:, 1:]
    if remove_end_token:
        ak_tokens = ak_tokens[:, :-1]
    if shift_tokens_minus_one:
        # if the tokens are in a nested structure, we need to shift all of them
        # by -1 separately, otherwise we can just shift the whole array
        ak_tokens = (
            ak_tokens - 1
            if len(ak_tokens.fields) == 0
            else ak.Array({field: ak_tokens[field] - 1 for field in ak_tokens.fields})
        )

    if load_token_ids:
        if len(ak_tokens.fields) == 0:
            if class_token_dict is not None:
                logger.info("Shifting tokens to accommodate class tokens")
                ak_tokens = ak.concatenate(
                    [
                        ak_tokens[:, :1],
                        class_token_array,
                        ak_tokens[:, 1:] + number_of_jettypes,
                    ],
                    axis=1,
                )
            # non-group tokenization (initial setup)
            x_ak_tokens = ak.Array(
                {
                    "part_token_id": ak_tokens,
                    "part_token_id_duplicated": ak_tokens,
                    "part_token_id_without_last": ak_tokens[:, :-1],
                    "part_token_id_without_first": ak_tokens[:, 1:],
                }
            )
        else:
            # group tokenization (new setup)
            dict_for_x_ak_tokens = {}
            for field in ak_tokens.fields:
                ak_tokens_last_with_stop_overwritten = ak.concatenate(
                    [
                        ak_tokens[field][:, :-1],
                        ak.ones_like(ak_tokens[field][:, -1:]) * stop_token_value_dict[field],
                    ],
                    axis=1,
                )
                ak_tokens_with_last_two_particles_with_stop_overwritten = ak.concatenate(
                    [
                        ak_tokens[field][:, :-2],
                        ak.ones_like(ak_tokens[field][:, -2:]) * stop_token_value_dict[field],
                    ],
                    axis=1,
                )
                dict_for_x_ak_tokens[field] = ak_tokens[field]
                dict_for_x_ak_tokens[f"{field}_last_with_stop_overwritten"] = (
                    ak_tokens_last_with_stop_overwritten
                )
                dict_for_x_ak_tokens[f"{field}_last_two_with_stop_overwritten"] = (
                    ak_tokens_with_last_two_particles_with_stop_overwritten
                )
                dict_for_x_ak_tokens[f"{field}_without_last"] = ak_tokens[field][:, :-1]
                dict_for_x_ak_tokens[f"{field}_without_first"] = ak_tokens[field][:, 1:]
                # duplicate of default token ids
            dict_for_x_ak_tokens["part_token_id_duplicated"] = ak_tokens["part_token_id"]
            x_ak_tokens = ak.Array(dict_for_x_ak_tokens)
        logger.info(f"Available fields in x_ak_tokens: {x_ak_tokens.fields}")
    else:
        x_ak_tokens = None

    # check if any other features than the token-id are requested
    if particle_features is not None:
        logger.info(
            "Loading the following features from the `particle_features` section of the "
            f"file: {particle_features}"
        )
        # we support both "particle_features" and "features" as key
        key_tmp = (
            "particle_features" if "particle_features" in tokenized_file.fields else "features"
        )
        x_ak_features = safe_load_features_from_ak_array(
            ak_array=tokenized_file[key_tmp][:n_load],
            features=particle_features,
            load_zeros_if_not_present=True,
        )
        if add_padded_particle_features_start or add_padded_particle_features_end:
            x_ak_features = ak_add_zero_padded_features(
                x_ak_features,
                add_start=add_padded_particle_features_start,
                add_end=add_padded_particle_features_end,
            )
    else:
        x_ak_features = None

    if particle_features_tokenized is not None:
        logger.info(
            "Loading the following features from the `particle_features_tokenized` section of "
            f"the file: {particle_features_tokenized}"
        )
        # we support both "particle_features_tokenized" and "features_tokenized" as key
        key_tmp = (
            "particle_features_tokenized"
            if "particle_features_tokenized" in tokenized_file.fields
            else "features_tokenized"
        )
        x_ak_features_tokenized = safe_load_features_from_ak_array(
            ak_array=tokenized_file[key_tmp][:n_load],
            features=particle_features_tokenized,
            load_zeros_if_not_present=True,
        )
        if add_padded_particle_features_start or add_padded_particle_features_end:
            x_ak_features_tokenized = ak_add_zero_padded_features(
                x_ak_features_tokenized,
                add_start=add_padded_particle_features_start,
                add_end=add_padded_particle_features_end,
            )
    else:
        x_ak_features_tokenized = None

    if jet_features is not None:
        pass
    else:
        x_jets = None

    # apply shuffling
    if random_seed is not None:
        logger.info(f"Shuffling data within one (tokenized) file with random seed {random_seed}.")
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(x_ak_tokens))
        x_ak_tokens = x_ak_tokens[permutation]
        x_ak_features = x_ak_features[permutation] if particle_features is not None else None
        x_ak_features_tokenized = (
            x_ak_features_tokenized[permutation]
            if particle_features_tokenized is not None
            else None
        )
        x_jets = x_jets[permutation] if jet_features is not None else None
        labels_onehot = labels_onehot[permutation]

    return (
        combine_ak_arrays(x_ak_tokens, x_ak_features, x_ak_features_tokenized),
        labels_onehot[labels],
        x_jets,
    )


def read_jetclass_file(
    filepath,
    particle_features=["part_pt", "part_eta", "part_phi", "part_energy"],
    jet_features=["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
    return_p4=False,
    n_load=None,
    shuffle_particles=False,
    random_seed: int = None,
):
    """Loads a single file from the JetClass dataset.

    Parameters
    ----------
    filepath : str
        Path to the ROOT data file.
    particle_features : List[str], optional
        A list of particle-level features to be loaded.
        Possible options are:
        - part_px
        - part_py
        - part_pz
        - part_energy
        - part_deta
        - part_dphi
        - part_d0val
        - part_d0err
        - part_dzval
        - part_dzerr
        - part_charge
        - part_isChargedHadron
        - part_isNeutralHadron
        - part_isPhoton
        - part_isElectron
        - part_isMuon

    jet_features : List[str], optional
        A list of jet-level features to be loaded.
        Possible options are:
        - jet_pt
        - jet_eta
        - jet_phi
        - jet_energy
        - jet_nparticles
        - jet_sdmass
        - jet_tau1
        - jet_tau2
        - jet_tau3
        - jet_tau4
        - aux_genpart_eta
        - aux_genpart_phi
        - aux_genpart_pid
        - aux_genpart_pt
        - aux_truth_match

    labels : List[str], optional
        A list of truth labels to be loaded.
        - label_QCD
        - label_Hbb
        - label_Hcc
        - label_Hgg
        - label_H4q
        - label_Hqql
        - label_Zqq
        - label_Wqq
        - label_Tbqq
        - label_Tbl

    return_p4 : bool, optional
        Whether to return the 4-momentum of the particles.
    n_load : int, optional
        Number of jets to load. If None, all jets are loaded.
    shuffle_particles : bool, optional
        If True, the particles are shuffled. Default is False.
    random_seed : int, optional
        Random seed for shuffling the jets. If None, no shuffling is performed.

    Returns
    -------
    x_particles : ak.Array
        An awkward array of the particle-level features.
    x_jets : ak.Array
        An awkward array of the jet-level features.
    y : ak.Array
        An awkward array of the truth labels (one-hot encoded).
    p4 : ak.Array, optional
        An awkward array of the 4-momenta of the particles. Only returned if
        `return_p4` is set to True.
    """

    if n_load is not None:
        table = uproot.open(filepath)["tree"].arrays()[:n_load]
    else:
        table = uproot.open(filepath)["tree"].arrays()

    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
            # massless particles -> this changes the result slightly,
            # i.e. for example top jets then have a mass of 171.2 instead of 172
            # "mass": ak.zeros_like(table["part_px"]),
        }
    )
    p4_jet = ak.sum(p4, axis=1)

    table["part_pt"] = p4.pt
    table["part_eta"] = p4.eta
    table["part_phi"] = p4.phi
    table["part_mass"] = p4.mass
    table["part_ptrel"] = table["part_pt"] / p4_jet.pt
    table["part_erel"] = table["part_energy"] / p4_jet.energy
    table["part_etarel"] = p4.deltaeta(p4_jet)
    table["part_phirel"] = p4.deltaphi(p4_jet)
    table["part_deltaR"] = p4.deltaR(p4_jet)
    table["jet_mass_from_p4s"] = p4_jet.mass
    table["jet_pt_from_p4s"] = p4_jet.pt
    table["jet_eta_from_p4s"] = p4_jet.eta
    table["jet_phi_from_p4s"] = p4_jet.phi
    table["part_p"] = p4.p
    # Add the energy a second time:
    # workaround to select this feature twice, which is e.g. the case in ParT, where
    # the particle energy is used as log(energy) as standard particle feature
    # and as raw energy in the lorentz vector features
    table["part_energy_raw"] = table["part_energy"]

    p4_centered = ak.zip(
        {
            "pt": p4.pt,
            "eta": p4.deltaeta(p4_jet),
            "phi": p4.deltaphi(p4_jet),
            "mass": p4.mass,
        },
        with_name="Momentum4D",
    )

    table["part_px_after_centering"] = p4_centered.px
    table["part_py_after_centering"] = p4_centered.py
    table["part_pz_after_centering"] = p4_centered.pz
    table["part_energy_after_centering_raw"] = p4_centered.energy

    # check if any of the requested features contains "massless"
    if any("massless" in feature for feature in particle_features):
        p4_massless = ak.zip(
            {
                "pt": p4.pt,
                "eta": p4.eta,
                "phi": p4.phi,
                "mass": ak.zeros_like(table["part_px"]),
            },
            with_name="Momentum4D",
        )
        p4_jet_massless = ak.sum(p4_massless, axis=1)
        # features corresponding to
        table["part_pt_massless"] = p4_massless.pt
        table["part_px_massless"] = p4_massless.px
        table["part_py_massless"] = p4_massless.py
        table["part_pz_massless"] = p4_massless.pz
        table["part_ptrel_massless"] = p4_massless.pt / p4_jet_massless.pt
        table["part_energy_massless"] = p4_massless.energy
        table["part_erel_massless"] = p4_massless.energy / p4_jet_massless.energy
        table["part_etarel_massless"] = p4_massless.deltaeta(p4_jet_massless)
        table["part_phirel_massless"] = p4_massless.deltaphi(p4_jet_massless)
        table["part_deltaR_massless"] = p4_massless.deltaR(p4_jet_massless)
        table["part_energy_massless_raw"] = p4_massless.energy

    # check if any of the requested features contains "mlc" (massless + centered)
    if any(("mlc" in feature or "massless_centered" in feature) for feature in particle_features):
        # refer to this below as "mlc"
        p4_massless_centered = ak.zip(
            {
                "pt": p4_massless.pt,
                "eta": p4_massless.deltaeta(p4_jet_massless),
                "phi": p4_massless.deltaphi(p4_jet_massless),
                "mass": ak.zeros_like(p4_massless.mass),
            },
            with_name="Momentum4D",
        )
        p4_jet_massless_centered = ak.sum(p4_massless_centered, axis=1)
        # features corresponding to using massless + centered particles ("mlc")
        table["part_energy_mlc"] = p4_massless_centered.energy
        table["part_px_mlc"] = p4_massless_centered.px
        table["part_py_mlc"] = p4_massless_centered.py
        table["part_pz_mlc"] = p4_massless_centered.pz
        table["part_energy_mlc"] = p4_massless_centered.energy
        table["part_energy_mlc_raw"] = p4_massless_centered.energy
        table["part_erel_mlc"] = p4_massless_centered.energy / p4_jet_massless_centered.energy
        table["part_ptrel_mlc"] = p4_massless.pt / p4_jet_massless_centered.pt
        table["part_px_massless_centered"] = p4_massless_centered.px
        table["part_py_massless_centered"] = p4_massless_centered.py
        table["part_pz_massless_centered"] = p4_massless_centered.pz
        table["part_energy_massless_centered"] = p4_massless_centered.energy

    x_particles = table[particle_features] if particle_features is not None else None
    if shuffle_particles:
        logger.info("Shuffling particles within each jet.")
        x_particles = shuffle_ak_arr_along_axis1(x_particles, seed=100)
    x_jets = table[jet_features] if jet_features is not None else None
    y = ak.values_astype(table[labels], "int32") if labels is not None else None

    if random_seed is not None:
        logger.info(f"Shuffling data (within one file) with random seed {random_seed}.")
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(x_particles))
        x_particles = x_particles[permutation] if x_particles is not None else None
        x_jets = x_jets[permutation] if x_jets is not None else None
        y = y[permutation] if y is not None else None
        p4 = p4[permutation] if return_p4 else None

    if return_p4:
        if shuffle_particles:
            raise ValueError("Cannot shuffle particles and return 4-momenta at the same time.")
        return x_particles, x_jets, y, p4

    return x_particles, x_jets, y


def load_landscape_file_and_split_into_qcd_and_top(
    filename: str,
    destination_folder: str,
    number: int,
    convert_to_jetclass_style_root: bool = False,
):
    """Will load the specified file, split it into qcd and tbqq and save them
    in the destination folder.
    The destination folder will get two files: ZJetsToNuNu.parquet and TTbar.parquet.
    This is useful to create the files for the landscape dataset in a JetClass-like format.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    destination_folder : str
        The path to the folder where the files should be saved.
    number : int
        The number to append to the filename.
    convert_to_jetclass_style_root : bool
        If the file should be converted to the JetClass style (i.e. root file with the
        same feature naming conventions).
    """
    ak_arr = ak.from_parquet(filename)
    is_qcd = ak_arr["label"] == 0
    # save the files
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving qcd and top to {destination_folder}")
    ak.to_parquet(ak_arr[is_qcd], destination_folder / f"ZJetsToNuNu_{number}.parquet")
    ak.to_parquet(ak_arr[~is_qcd], destination_folder / f"TTBar_{number}.parquet")

    if convert_to_jetclass_style_root:
        logger.info("Converting to JetClass style root files.")
        landscape_file_to_jetclass_style(
            destination_folder / f"ZJetsToNuNu_{number}.parquet",
            destination_folder / f"ZJetsToNuNu_{number}.root",
        )
        landscape_file_to_jetclass_style(
            destination_folder / f"TTBar_{number}.parquet",
            destination_folder / f"TTBar_{number}.root",
        )
        logger.info("Deleting parquet files.")
        (destination_folder / f"ZJetsToNuNu_{number}.parquet").unlink()
        (destination_folder / f"TTBar_{number}.parquet").unlink()


def load_lhco_jets_from_parquet(
    parquet_filename,
    feature_dict: dict,
    n_jets: int = None,
):
    """Helper function to load LHCO jets from parquet file.

    Parameters
    ----------
    parquet_filename : str
        Path to parquet file. This should have the following structure:
        n_events * var * Momentum4D[
            pt: float64,
            eta: float64,
            phi: float64,
            mass: float64
        ]
    feature_dict : list
        Dictionary of features to load from the parquet file and their preprocessing
        parameters
    n_jets : int
        Number of jets to load from the parquet file

    Returns
    -------
    ak.Array
        ak.Array of jet constituent features
    ak.Array
        ak.Array of jet labels (0 for QCD, 1 for top)
    """

    batch_size = 100_000
    ak_arr = ak.from_parquet(parquet_filename)

    if n_jets is None:
        n_jets = len(ak_arr)

    n_batches = (n_jets + batch_size - 1) // batch_size

    all_particle_features_list = []
    labels_list = []

    # Loop over batches and calculate all possible features
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_jets)
        batch_ak_arr = ak_arr[start_idx:end_idx]
        batch_p4 = ak.zip(
            {
                "pt": batch_ak_arr["pt"],
                "eta": batch_ak_arr["eta"],
                "phi": batch_ak_arr["phi"],
                "mass": batch_ak_arr["mass"],
            },
            with_name="Momentum4D",
        )

        jets = ak.sum(batch_p4, axis=1)

        batch_particle_features = ak.Array(
            {
                "part_pt": batch_p4.pt,
                "part_eta": batch_p4.eta,
                "part_phi": batch_p4.phi,
                "part_energy": batch_p4.energy,
                "part_energy_raw": batch_p4.energy,
                "part_erel": batch_p4.energy / jets.energy,
                "part_etarel": batch_p4.deltaeta(jets),
                "part_phirel": batch_p4.deltaphi(jets),
                "part_ptrel": batch_p4.pt / jets.pt,
                "part_deltaR": batch_p4.deltaR(jets),
                "part_px": batch_p4.px,
                "part_py": batch_p4.py,
                "part_pz": batch_p4.pz,
                "part_mass": batch_p4.mass,
            }
        )
        all_particle_features_list.append(batch_particle_features)
        # labels_list.append(batch_ak_arr["label"])
        # use label=8 for top jets and label=0 for QCD jets (as in JetClass)
        # labels_list.append(8 * (batch_ak_arr["label"] == 1))
        labels_list.append(np.ones(len(batch_particle_features)))

    all_particle_features = ak.concatenate(all_particle_features_list)
    labels_ones = np.concatenate(labels_list)
    labels_zeros = labels_ones - 1

    if "ZJets" in parquet_filename:
        labels = ak.Array(
            {"label_QCD": labels_ones, "label_WprimeToXY": labels_zeros}
        )  # For now everything is QCD
    elif "Wprime" in parquet_filename:
        labels = ak.Array({"label_QCD": labels_zeros, "label_WprimeToXY": labels_ones})
    else:
        raise ValueError("Files need to be either ZJetsToNuNu or WprimeToXY")

    return ak_select_and_preprocess(all_particle_features, pp_dict=feature_dict), labels

def load_lhco_jets_from_h5(
    h5_filename,
    feature_dict: dict,
    n_jets: int = None,
    jet_name: str = "jet1", # Options: "jet1", "jet2", "both"
    mom4_format: str = "epxpypz",  # Options: "epxpypz", "pxpypze", "ptphietam"
    use_h5_features: bool = True,
):
    """Load LHCO jets from HDF5 file and convert to awkward array format.

    Parameters
    ----------
    h5_filename : str
        Path to HDF5 file with structure:
        /jet1/4mom: (n_events, n_particles, 4) - 4-momentum of jet1 constituents
        /jet1/coords: (n_events, n_particles, 2) - absolute (eta, phi) coordinates
        /jet1/features: (n_events, n_particles, 9) - precomputed features:
            [eta_rel, phi_rel, log(pt), log(e), pt/pt_jet, e/e_jet, 
             log(pt/pt_jet), log(e/e_jet), deltaR]
            Note: eta_rel and phi_rel are RELATIVE to jet axis, not absolute!
        /jet1/mask: (n_events, n_particles, 1) - constituent mask (1=real, 0=padding)
        /jet2/4mom: (n_events, n_particles, 4) - 4-momentum of jet2 constituents
        /jet2/coords: (n_events, n_particles, 2) - absolute (eta, phi) coordinates
        /jet2/features: (n_events, n_particles, 9) - precomputed features (same structure as jet1)
        /jet2/mask: (n_events, n_particles, 1) - constituent mask
        /jet_coords: (n_events, 2, 4) - jet-level (pt, eta, phi, m) for both jets
        /jet_features: (n_events, 7) - (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)
        /signal: (n_events,) - binary labels (0=background, 1=signal)
    feature_dict : dict
        Dictionary of features to load and their preprocessing parameters.
    n_jets : int, optional
        Number of jets to load from the file. If None, load all jets.
    jet_name : str, optional
        Name of the jet group in HDF5 file:
        - "jet1": Load only jet1 constituents
        - "jet2": Load only jet2 constituents  
        - "both": Load both jets separately
        Default: "jet1"
    mom4_format : str, optional
        Format of the 4-momentum in the HDF5 file:
        - "epxpypz": [E, px, py, pz] (energy first)
        - "pxpypze": [px, py, pz, E] (energy last)
        - "ptphietam": [pt, phi, eta, mass]
        Default: "epxpypz"
        Note: Only used when use_h5_features=False
    use_h5_features : bool, optional
        If True, use the precomputed features from /jet1/features dataset (recommended).
        If False, derive features from 4-momentum (slower, may have numerical differences).
        Default: True

    Returns
    -------
    When jet_name="jet1" or "jet2":
        preprocessed_features : ak.Array
            Awkward array of jet constituent features (after preprocessing)
        labels : np.ndarray
            Numpy array of jet labels (0=background, 1=signal)
    
    When jet_name="both":
        preprocessed_features_jet1 : ak.Array
            Awkward array of jet1 constituent features (after preprocessing)
        preprocessed_features_jet2 : ak.Array
            Awkward array of jet2 constituent features (after preprocessing)
        labels : np.ndarray
            Numpy array of jet labels (0=background, 1=signal)
    """
    # Handle "both" case by loading jet1 and jet2 separately
    if jet_name == "both":
        jet1_features, labels = load_lhco_jets_from_h5(
            h5_filename=h5_filename,
            feature_dict=feature_dict,
            n_jets=n_jets,
            jet_name="jet1",
            mom4_format=mom4_format,
            use_h5_features=use_h5_features,
        )
        jet2_features, _ = load_lhco_jets_from_h5(
            h5_filename=h5_filename,
            feature_dict=feature_dict,
            n_jets=n_jets,
            jet_name="jet2",
            mom4_format=mom4_format,
            use_h5_features=use_h5_features,
        )
        return jet1_features, jet2_features, labels

    batch_size = 10000

    with h5py.File(h5_filename, "r") as f:
        # Get total number of jets
        total_jets = f[f"{jet_name}/4mom"].shape[0]
        if n_jets is None:
            n_jets = total_jets
        else:
            n_jets = min(n_jets, total_jets)

        n_batches = (n_jets + batch_size - 1) // batch_size

        all_particle_features_list = []
        labels_list = []

        # Loop over batches
        for i in tqdm(range(n_batches), desc=f"Loading {os.path.basename(h5_filename)} ({jet_name})"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_jets)

            # Load batch data
            batch_mask = f[f"{jet_name}/mask"][start_idx:end_idx].squeeze(-1)  # (batch, n_particles)
            batch_labels = f["signal"][start_idx:end_idx]  # (batch,)
            
            # Use precomputed features if available and requested
            if use_h5_features and f"{jet_name}/features" in f:
                batch_h5_features = f[f"{jet_name}/features"][start_idx:end_idx]  # (batch, n_particles, 9)
                batch_coords = f[f"{jet_name}/coords"][start_idx:end_idx]  # (batch, n_particles, 2)
                batch_4mom = f[f"{jet_name}/4mom"][start_idx:end_idx]  # (batch, n_particles, 4)
                
                # Get jet-level coordinates to convert relative features to absolute if needed
                if "jet_coords" in f:
                    batch_jet_coords = f["jet_coords"][start_idx:end_idx]  # (batch, 2, 4)
                    # Extract jet1 or jet2 based on jet_name
                    jet_idx = 0 if jet_name == "jet1" else 1
                    batch_jet_pt = batch_jet_coords[:, jet_idx, 0]
                    batch_jet_eta = batch_jet_coords[:, jet_idx, 1]
                    batch_jet_phi = batch_jet_coords[:, jet_idx, 2]
                    batch_jet_m = batch_jet_coords[:, jet_idx, 3]
                
                # Convert to jagged arrays using mask
                batch_h5_features_jagged = ak.Array(
                    [batch_h5_features[i][batch_mask[i].astype(bool)] for i in range(len(batch_h5_features))]
                )
                batch_coords_jagged = ak.Array(
                    [batch_coords[i][batch_mask[i].astype(bool)] for i in range(len(batch_coords))]
                )
                batch_4mom_jagged = ak.Array(
                    [batch_4mom[i][batch_mask[i].astype(bool)] for i in range(len(batch_4mom))]
                )
                
                # Parse 4-momentum to get energy and pt for features not in h5
                if mom4_format == "epxpypz":
                    energy = batch_4mom_jagged[:, :, 0]
                    px = batch_4mom_jagged[:, :, 1]
                    py = batch_4mom_jagged[:, :, 2]
                    pz = batch_4mom_jagged[:, :, 3]
                elif mom4_format == "pxpypze":
                    px = batch_4mom_jagged[:, :, 0]
                    py = batch_4mom_jagged[:, :, 1]
                    pz = batch_4mom_jagged[:, :, 2]
                    energy = batch_4mom_jagged[:, :, 3]
                elif mom4_format == "ptphietam":
                    # Create p4 from stored pt, eta, phi, mass
                    batch_p4_temp = ak.zip(
                        {
                            "pt": batch_4mom_jagged[:, :, 0],
                            "phi": batch_4mom_jagged[:, :, 1],
                            "eta": batch_4mom_jagged[:, :, 2],
                            "mass": batch_4mom_jagged[:, :, 3],
                        },
                        with_name="Momentum4D",
                    )
                    px = batch_p4_temp.px
                    py = batch_p4_temp.py
                    pz = batch_p4_temp.pz
                    energy = batch_p4_temp.energy
                else:
                    raise ValueError(
                        f"Unknown mom4_format: {mom4_format}. "
                        "Use 'epxpypz', 'pxpypze', or 'ptphietam'"
                    )
                
                pt = np.sqrt(px**2 + py**2)
                eta = batch_coords_jagged[:, :, 0]  # absolute eta from coords
                phi = batch_coords_jagged[:, :, 1]  # absolute phi from coords
                mass = np.sqrt(np.maximum(0, energy**2 - px**2 - py**2 - pz**2))
                
                # Create Momentum4D for deltaR and other vector operations
                batch_p4 = ak.zip(
                    {
                        "pt": pt,
                        "eta": eta,
                        "phi": phi,
                        "mass": mass,
                    },
                    with_name="Momentum4D",
                )
                
                # Reconstruct jet 4-momentum
                jets = ak.sum(batch_p4, axis=1)
                
                # Build feature dictionary using precomputed features where available
                # h5_features: [eta_rel, phi_rel, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR]
                batch_particle_features = ak.Array(
                    {
                        "part_pt": pt,
                        "part_eta": eta,  # absolute
                        "part_phi": phi,  # absolute
                        "part_energy": energy,
                        "part_energy_raw": energy,
                        "part_erel": batch_h5_features_jagged[:, :, 5],  # from h5
                        "part_etarel": batch_h5_features_jagged[:, :, 0],  # from h5
                        "part_phirel": batch_h5_features_jagged[:, :, 1],  # from h5
                        "part_ptrel": batch_h5_features_jagged[:, :, 4],  # from h5
                        "part_deltaR": batch_h5_features_jagged[:, :, 8],  # from h5
                        "part_px": px,
                        "part_py": py,
                        "part_pz": pz,
                        "part_mass": mass,
                        # Additional features that might be requested
                        "part_log_pt": batch_h5_features_jagged[:, :, 2],  # from h5
                        "part_log_energy": batch_h5_features_jagged[:, :, 3],  # from h5
                        "part_log_ptrel": batch_h5_features_jagged[:, :, 6],  # from h5
                        "part_log_erel": batch_h5_features_jagged[:, :, 7],  # from h5
                    }
                )
                
            else:
                # Fallback to computing features from 4-momentum
                batch_4mom = f[f"{jet_name}/4mom"][start_idx:end_idx]  # (batch, n_particles, 4)
                
                # Convert to awkward arrays and apply mask
                # Create jagged arrays based on mask
                batch_4mom_jagged = ak.Array(
                    [batch_4mom[i][batch_mask[i].astype(bool)] for i in range(len(batch_4mom))]
                )

                # Parse 4-momentum based on format
                if mom4_format == "epxpypz":
                    # [E, px, py, pz]
                    energy = batch_4mom_jagged[:, :, 0]
                    px = batch_4mom_jagged[:, :, 1]
                    py = batch_4mom_jagged[:, :, 2]
                    pz = batch_4mom_jagged[:, :, 3]
                elif mom4_format == "pxpypze":
                    # [px, py, pz, E]
                    px = batch_4mom_jagged[:, :, 0]
                    py = batch_4mom_jagged[:, :, 1]
                    pz = batch_4mom_jagged[:, :, 2]
                    energy = batch_4mom_jagged[:, :, 3]
                elif mom4_format == "ptphietam":
                    # [pt, phi, eta, mass] - create Momentum4D directly
                    batch_p4 = ak.zip(
                        {
                            "pt": batch_4mom_jagged[:, :, 0],
                            "phi": batch_4mom_jagged[:, :, 1],
                            "eta": batch_4mom_jagged[:, :, 2],
                            "mass": batch_4mom_jagged[:, :, 3],
                        },
                        with_name="Momentum4D",
                    )
                    # Create px, py, pz, energy for backup
                    px = batch_p4.px
                    py = batch_p4.py
                    pz = batch_p4.pz
                    energy = batch_p4.energy
                else:
                    raise ValueError(
                        f"Unknown mom4_format: {mom4_format}. "
                        "Use 'epxpypz', 'pxpypze', or 'ptphietam'"
                    )

                # Create Momentum4D from Cartesian coordinates if not already created
                if mom4_format != "ptphietam":
                    # Calculate pt, eta, phi, mass from px, py, pz, energy
                    pt = np.sqrt(px**2 + py**2)
                    eta = np.arctanh(pz / np.sqrt(px**2 + py**2 + pz**2))
                    phi = np.arctan2(py, px)
                    mass = np.sqrt(np.maximum(0, energy**2 - px**2 - py**2 - pz**2))

                    batch_p4 = ak.zip(
                        {
                            "pt": pt,
                            "eta": eta,
                            "phi": phi,
                            "mass": mass,
                        },
                        with_name="Momentum4D",
                    )

                # Calculate jet-level 4-momentum (sum over particles)
                jets = ak.sum(batch_p4, axis=1)

                # Create feature dictionary similar to load_lhco_jets_from_parquet
                batch_particle_features = ak.Array(
                    {
                        "part_pt": batch_p4.pt,
                        "part_eta": batch_p4.eta,
                        "part_phi": batch_p4.phi,
                        "part_energy": batch_p4.energy,
                        "part_energy_raw": batch_p4.energy,
                        "part_erel": batch_p4.energy / jets.energy,
                        "part_etarel": batch_p4.deltaeta(jets),
                        "part_phirel": batch_p4.deltaphi(jets),
                        "part_ptrel": batch_p4.pt / jets.pt,
                        "part_deltaR": batch_p4.deltaR(jets),
                        "part_px": batch_p4.px,
                        "part_py": batch_p4.py,
                        "part_pz": batch_p4.pz,
                        "part_mass": batch_p4.mass,
                    }
                )

            all_particle_features_list.append(batch_particle_features)
            labels_list.append(batch_labels)

    # Concatenate all batches
    all_particle_features = ak.concatenate(all_particle_features_list)
    all_labels = np.concatenate(labels_list)

    # Use binary labels directly from HDF5 file
    # signal=1 for signal jets, signal=0 for background jets
    labels = np.array(all_labels, dtype=int)

    # Apply preprocessing and feature selection
    preprocessed_features = ak_select_and_preprocess(
        all_particle_features, pp_dict=feature_dict
    )

    return preprocessed_features, labels


def load_multiple_h5_files(
    h5_filenames: list,
    feature_dict: dict,
    n_jets_per_file: int | list = None,
    mom4_format: str = "epxpypz",
    jet_name: str = "jet1", # Options: "jet1", "jet2", "both"
    **kwargs,
):
    """Load and concatenate multiple HDF5 files.

    Parameters
    ----------
    h5_filenames : list of str
        List of paths to HDF5 files
    feature_dict : dict
        Dictionary of features and preprocessing parameters
    n_jets_per_file : int, optional
        Number of jets to load from each file. If None, load all.
    **kwargs
        Additional arguments passed to load_lhco_jets_from_h5

    Returns
    -------
    ak.Array
        Combined particle features
    ak.Array
        Combined labels
    """
    all_jet1_features = []
    all_jet2_features = []
    all_features = []
    all_labels = []

    # Handle n_jets_per_file as int or list
    if n_jets_per_file is None:
        n_jets_list = [None] * len(h5_filenames)
    elif isinstance(n_jets_per_file, int):
        n_jets_list = [n_jets_per_file] * len(h5_filenames)
    elif isinstance(n_jets_per_file, list):
        if len(n_jets_per_file) != len(h5_filenames):
            raise ValueError(
                f"Length of n_jets_per_file ({len(n_jets_per_file)}) "
                f"must match length of h5_filenames ({len(h5_filenames)})"
            )
        n_jets_list = n_jets_per_file
    else:
        raise TypeError(
            f"n_jets_per_file must be int, list, or None, got {type(n_jets_per_file)}"
        )
    
    if jet_name == "both":
        for filename, n_jets in zip(h5_filenames, n_jets_list):
            jet1_features, jet2_features, labels = load_lhco_jets_from_h5(
                h5_filename=filename,
                feature_dict=feature_dict,
                n_jets=n_jets,
                jet_name=jet_name,
                mom4_format=mom4_format,
                **kwargs
            )
            all_jet1_features.append(jet1_features)
            all_jet2_features.append(jet2_features)
            all_labels.append(labels)

        # Concatenate all files
        combined_features_jet1 = ak.concatenate(all_jet1_features)
        combined_features_jet2 = ak.concatenate(all_jet2_features)
        # Concatenate labels (they are numpy arrays now)
        combined_labels = np.concatenate(all_labels)

        return combined_features_jet1, combined_features_jet2, combined_labels
    
    elif jet_name in ["jet1", "jet2"]:
        for filename, n_jets in zip(h5_filenames, n_jets_list):
            features, labels = load_lhco_jets_from_h5(
                h5_filename=filename,
                feature_dict=feature_dict,
                n_jets=n_jets,
                jet_name=jet_name,
                mom4_format=mom4_format,
                **kwargs
            )
            all_features.append(features)
            all_labels.append(labels)

        # Concatenate all files
        combined_features = ak.concatenate(all_features)
        # Concatenate labels (they are numpy arrays now)
        combined_labels = np.concatenate(all_labels)

        return combined_features, combined_labels


def load_case_jets_from_h5(
    h5_filename,
    feature_dict: dict,
    n_jets: int = None,
    jet_name: str = "jet1",  # Options: "jet1", "jet2", "both"
):
    """Load CASE dijet events from an HDF5 file.

    Parameters
    ----------
    h5_filename : str
        Path to a CASE HDF5 file.  Both background and signal files share the
        same per-jet keys:

        - jet1_PFCands / jet2_PFCands : (N, 100, 4) float16
          Columns are **[px, py, pz, E]** for each PF candidate.
          Padding particles are all-zero.
        - jet1_extraInfo / jet2_extraInfo : (N, 7) float32
          Columns: [tau1, tau2, tau3, tau4, btag_score, ?, nPFCands]
        - jet_kinematics : (N, 14) float32
          Columns: [m_jj, delta_eta,
                    j1_pt, j1_eta, j1_phi, j1_mass,
                    j2_pt, j2_eta, j2_phi, j2_mass,
                    j3_pt, j3_eta, j3_phi, j3_mass]
        - truth_label : (N, 1) int  –  0 = background, 1 = signal

    feature_dict : dict
        Feature selection and preprocessing parameters.  Available keys:
        part_pt, part_eta, part_phi, part_energy,
        part_etarel, part_phirel, part_ptrel, part_erel,
        part_deltaR, part_log_pt, part_log_energy,
        part_log_ptrel, part_log_erel,
        part_px, part_py, part_pz, part_mass.
    n_jets : int, optional
        Number of events to load.  None loads everything.
    jet_name : str, optional
        "jet1", "jet2", or "both".

    Returns
    -------
    When jet_name is "jet1" or "jet2"
        preprocessed_features : ak.Array
        labels : np.ndarray  (0 = background, 1 = signal)

    When jet_name is "both"
        jet1_features, jet2_features : ak.Array
        labels : np.ndarray
    """
    if jet_name == "both":
        jet1_features, labels = load_case_jets_from_h5(
            h5_filename=h5_filename,
            feature_dict=feature_dict,
            n_jets=n_jets,
            jet_name="jet1",
        )
        jet2_features, _ = load_case_jets_from_h5(
            h5_filename=h5_filename,
            feature_dict=feature_dict,
            n_jets=n_jets,
            jet_name="jet2",
        )
        return jet1_features, jet2_features, labels

    pf_key = "jet1_PFCands" if jet_name == "jet1" else "jet2_PFCands"
    batch_size = 10000

    with h5py.File(h5_filename, "r") as f:
        total_jets = f[pf_key].shape[0]
        n_jets = total_jets if n_jets is None else min(n_jets, total_jets)
        n_batches = (n_jets + batch_size - 1) // batch_size

        all_particle_features_list = []
        labels_list = []

        for i in tqdm(
            range(n_batches),
            desc=f"Loading {os.path.basename(h5_filename)} ({jet_name})",
        ):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_jets)

            # (batch, 100, 4) float16 -> float32; columns are [px, py, pz, E]
            pf_cands = f[pf_key][start:end].astype(np.float32)
            labels_batch = f["truth_label"][start:end].squeeze(-1)  # (batch,)

            # Mask: real particles have at least one non-zero component
            mask = np.any(pf_cands != 0.0, axis=-1)  # (batch, 100)

            # Build jagged (variable-length) arrays by applying the mask
            px_jagged = ak.Array([pf_cands[j, mask[j], 0] for j in range(len(pf_cands))])
            py_jagged = ak.Array([pf_cands[j, mask[j], 1] for j in range(len(pf_cands))])
            pz_jagged = ak.Array([pf_cands[j, mask[j], 2] for j in range(len(pf_cands))])
            E_jagged  = ak.Array([pf_cands[j, mask[j], 3] for j in range(len(pf_cands))])

            # Four-momentum (vector library handles kinematics)
            p4 = ak.zip(
                {"px": px_jagged, "py": py_jagged, "pz": pz_jagged, "energy": E_jagged},
                with_name="Momentum4D",
            )
            jet_p4 = ak.sum(p4, axis=1)

            pt     = p4.pt
            eta    = p4.eta
            phi    = p4.phi
            energy = p4.energy
            mass   = p4.mass

            batch_feats = ak.Array(
                {
                    "part_pt":         pt,
                    "part_eta":        eta,
                    "part_phi":        phi,
                    "part_energy":     energy,
                    "part_energy_raw": energy,
                    "part_mass":       mass,
                    "part_px":         px_jagged,
                    "part_py":         py_jagged,
                    "part_pz":         pz_jagged,
                    "part_etarel":     p4.deltaeta(jet_p4),
                    "part_phirel":     p4.deltaphi(jet_p4),
                    "part_ptrel":      pt / jet_p4.pt,
                    "part_erel":       energy / jet_p4.energy,
                    "part_deltaR":     p4.deltaR(jet_p4),
                    "part_log_pt":     np.log(pt + 1e-8),
                    "part_log_energy": np.log(energy + 1e-8),
                    "part_log_ptrel":  np.log(pt / jet_p4.pt + 1e-8),
                    "part_log_erel":   np.log(energy / jet_p4.energy + 1e-8),
                }
            )

            all_particle_features_list.append(batch_feats)
            labels_list.append(labels_batch)

    all_particle_features = ak.concatenate(all_particle_features_list)
    all_labels = np.concatenate(labels_list).astype(int)

    preprocessed_features = ak_select_and_preprocess(
        all_particle_features, pp_dict=feature_dict
    )
    return preprocessed_features, all_labels


def load_multiple_case_h5_files(
    h5_filenames: list,
    feature_dict: dict,
    n_jets_per_file: int | list = None,
    jet_name: str = "jet1",  # Options: "jet1", "jet2", "both"
):
    """Load and concatenate multiple CASE HDF5 files.

    Parameters
    ----------
    h5_filenames : list of str
        Paths to CASE HDF5 files.
    feature_dict : dict
        Feature selection and preprocessing parameters.
    n_jets_per_file : int or list of int, optional
        Number of events per file.  If ``int``, the same limit is used for
        every file.  If ``list``, it must match *h5_filenames* in length.
        ``None`` loads all events from every file.
    jet_name : str, optional
        ``"jet1"``, ``"jet2"``, or ``"both"``.

    Returns
    -------
    See :func:`load_case_jets_from_h5` for return signatures.
    """
    if n_jets_per_file is None:
        n_jets_list = [None] * len(h5_filenames)
    elif isinstance(n_jets_per_file, int):
        n_jets_list = [n_jets_per_file] * len(h5_filenames)
    elif isinstance(n_jets_per_file, list):
        if len(n_jets_per_file) != len(h5_filenames):
            raise ValueError(
                f"Length of n_jets_per_file ({len(n_jets_per_file)}) "
                f"must match length of h5_filenames ({len(h5_filenames)})"
            )
        n_jets_list = n_jets_per_file
    else:
        raise TypeError(
            f"n_jets_per_file must be int, list, or None, got {type(n_jets_per_file)}"
        )

    if jet_name == "both":
        all_jet1, all_jet2, all_labels = [], [], []
        for fname, n_jets in zip(h5_filenames, n_jets_list):
            j1, j2, labels = load_case_jets_from_h5(
                fname, feature_dict, n_jets=n_jets, jet_name="both"
            )
            all_jet1.append(j1)
            all_jet2.append(j2)
            all_labels.append(labels)
        return (
            ak.concatenate(all_jet1),
            ak.concatenate(all_jet2),
            np.concatenate(all_labels),
        )
    elif jet_name in ["jet1", "jet2"]:
        all_feats, all_labels = [], []
        for fname, n_jets in zip(h5_filenames, n_jets_list):
            feats, labels = load_case_jets_from_h5(
                fname, feature_dict, n_jets=n_jets, jet_name=jet_name
            )
            all_feats.append(feats)
            all_labels.append(labels)
        return ak.concatenate(all_feats), np.concatenate(all_labels)


def create_mini_root_file(input_file, output_file, max_entries=1000):
    """
    Create a new root file with a subset of the entries from the input file.

    Parameters
    ----------
    input_file : str
        Path to the input root file.
    output_file : str
        Path to the output root file.
    max_entries : int (optional)
        Maximum number of entries to include in the output root file.
        Default is 1000.
    """
    # load the initial root file and create a new root file with a subset of the entries
    with uproot.open(input_file) as file:
        tree = file["tree"]
        ak_arrays = tree.arrays(library="ak", how="zip")[:max_entries]
        dict_for_writing = {key: ak_arrays[key] for key in ak_arrays.fields}
    with uproot.recreate(output_file) as file:
        file["tree"] = dict_for_writing
        print(f"Created {output_file} with {max_entries} entries")


def landscape_file_to_jetclass_style(filename, save_path, n_jets=None):
    """Convert a landscape file to the JetClass style.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    save_path : str
        The path to the file to save.
    """

    ak_arr = ak.from_parquet(filename)[:n_jets]
    labels_dict = {
        "label_Hbb": np.zeros(len(ak_arr)),
        "label_Hcc": np.zeros(len(ak_arr)),
        "label_Hgg": np.zeros(len(ak_arr)),
        "label_H4q": np.zeros(len(ak_arr)),
        "label_Hqql": np.zeros(len(ak_arr)),
        "label_Zqq": np.zeros(len(ak_arr)),
        "label_Wqq": np.zeros(len(ak_arr)),
        "label_Tbl": np.zeros(len(ak_arr)),
    }
    # check if it's a top jet file with two steps
    if "TTBar" in str(filename):
        # confirm that the file is a top jet file
        assert np.all(ak_arr["label"] == 1)
        labels_dict["label_Tbqq"] = np.ones(len(ak_arr))
        labels_dict["label_QCD"] = np.zeros(len(ak_arr))
    elif "ZJetsToNuNu" in str(filename):
        # confirm that the file is a qcd jet file
        assert np.all(ak_arr["label"] == 0)
        labels_dict["label_QCD"] = np.ones(len(ak_arr))
        labels_dict["label_Tbqq"] = np.zeros(len(ak_arr))
    else:
        raise ValueError(
            "File is not a top or qcd jet file. Two things are checked:"
            "1. The label is 1 for top jets and 0 for qcd jets."
            "2. The filename contains 'TTBar' for top jets and 'ZJetsToNuNu' for qcd jets."
        )

    p4s = ak.zip(
        {
            "px": ak_arr["part_px"],
            "py": ak_arr["part_py"],
            "pz": ak_arr["part_pz"],
            "energy": ak_arr["part_energy"],
        },
        with_name="Momentum4D",
    )
    jets = ak.sum(p4s, axis=1)

    particle_features = {
        "part_px": p4s.px,
        "part_py": p4s.py,
        "part_pz": p4s.pz,
        "part_energy": p4s.energy,
        "part_pt": p4s.pt,
        "part_eta": p4s.eta,
        "part_phi": p4s.phi,
        "part_deta": p4s.deltaeta(jets),
        "part_dphi": p4s.deltaphi(jets),
        "part_etarel": p4s.deltaeta(jets),
        "part_phirel": p4s.deltaphi(jets),
        "part_ptrel": p4s.pt / jets.pt,
        "part_erel": p4s.energy / jets.energy,
        "part_deltaR": p4s.deltaR(jets),
    }

    jet_features = {
        "jet_pt": jets.pt,
        "jet_eta": jets.eta,
        "jet_phi": jets.phi,
        "jet_energy": jets.energy,
        "jet_mass": jets.mass,
        "jet_nparticles": ak.num(p4s),
    }

    logger.info(f"Saving to {save_path}")

    with uproot.recreate(save_path) as f:
        f["tree"] = particle_features | jet_features | labels_dict


def safe_load_features_from_ak_array(
    ak_array: ak.Array,
    features: list,
    load_zeros_if_not_present: bool = False,
    verbose: bool = True,
) -> ak.Array:
    """Load features from an awkward array, checking if they are present.

    Parameters
    ----------
    ak_array : ak.Array
        The awkward array to load the features from.
    features : list
        List of features to load.
    load_zeros_if_not_present : bool, optional
        If True, load zeros for features that are not present. Default is False.
    verbose : bool, optional
        If True, print information about the features being loaded. Default is True.

    Returns
    -------
    ak.Array
        An awkward array with the requested features.
    """
    available_features = [f for f in features if f in ak_array.fields]
    if not available_features:
        raise ValueError("No requested features are present in the awkward array.")

    if verbose:
        logger.info(f"Loading features: {available_features} from the awkward array.")
        logger.info(f"Available features in the awkward array: {ak_array.fields}.")

    missing_features = set(features) - set(available_features)

    if load_zeros_if_not_present:
        if len(missing_features) != 0:
            logger.warning(
                f"Features {missing_features} are not present in the awkward array. "
                "Loading zeros for these features."
            )
        for feature in missing_features:
            ak_array[feature] = ak.zeros_like(ak_array[available_features[0]])
        return ak_array[features]
    else:
        logger.warning(
            f"Features {missing_features} are not present in the awkward array. "
            "They will not be loaded."
        )
        return ak_array[available_features]
