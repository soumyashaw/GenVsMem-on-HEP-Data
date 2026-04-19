"""Stuff to perform binning in the context of tokenization."""

import itertools
from typing import Union

import awkward as ak
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, np_to_ak
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


def apply_binning(
    arr: Union[np.ndarray, ak.Array],
    bin_edges: np.ndarray,
    return_bin_centers: bool = False,
    return_binned_values: bool = False,
) -> Union[
    ak.Array,
    tuple[ak.Array, np.ndarray],
    tuple[ak.Array, np.ndarray, ak.Array],
]:
    """Apply binning to an array.

    Parameters
    ----------
    arr : Union[np.ndarray, ak.Array]
        The array to be binned.
    bin_edges : np.ndarray
        The bin edges of the binning. Can be non-uniform.
    return_bin_centers : bool, optional
        Whether to return the bin centers, by default False
    return_binned_values : bool, optional
        Whether to return the binned values, by default False

    Returns
    -------
    Union[ak.Array, Tuple[ak.Array, np.ndarray], Tuple[ak.Array, np.ndarray, ak.Array]]
        The binned array, the bin centers (if return_bin_centers=True),
        and the bin indices (if return_binned_values=True).
        The order of the return values is always (bin_indices, bin_centers, binned_arr).
    """

    # if arr is an awkward array, convert it to a numpy array
    is_ak = isinstance(arr, (ak.Array, ak.highlevel.Array))
    if is_ak:
        ak_arr = arr
        arr = ak.to_numpy(ak.Array(ak_arr).layout.content)
    else:
        print("Warning: apply_binning was called with a non-awkward array.")
        print("type(arr):", type(arr))
        ak_arr = None

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # clip the values to the bin edges
    arr = np.clip(arr, bin_edges[0], bin_edges[-1])

    # compute the bin indices
    bin_indices = np.digitize(arr, bin_edges, right=False)
    # digitize returns 1-based indices, 0=underflow, n_bins+1=overflow
    # --> fix this to 0-based indices with under/overflow values set to 0/(n_bins-1)
    #     such that the left and rightmost bins are the under/overflow bins
    # set underflow to 1
    bin_indices = np.where(bin_indices == 0, 1, bin_indices)
    # set overflow to n_bins
    bin_indices = np.where(bin_indices == n_bins + 1, n_bins, bin_indices)
    # make indices 0-based
    bin_indices -= 1

    # Replace each bin index with the corresponding bin center
    binned_arr = bin_centers[bin_indices]

    if is_ak:
        binned_arr = ak.Array(
            ak.contents.ListOffsetArray(ak_arr.layout.offsets, ak.Array(binned_arr).layout)
        )
        bin_indices = ak.Array(
            ak.contents.ListOffsetArray(ak_arr.layout.offsets, ak.Array(bin_indices).layout)
        )
    if return_bin_centers and return_binned_values:
        return bin_indices, bin_centers, binned_arr
    elif return_binned_values:
        return bin_indices, binned_arr
    elif return_bin_centers:
        return bin_indices, bin_centers
    return bin_indices


class BinningTokenizer(nn.Module):
    def __init__(self, bin_edges_dict):
        super().__init__()
        self.bin_edges_dict = bin_edges_dict
        self.bin_centers_dict = {
            key: self.get_bin_centers(edges) for key, edges in bin_edges_dict.items()
        }

        logger.info("Bin edges:")
        for key, edges in bin_edges_dict.items():
            logger.info(
                f"{key} : leftmost={edges[0]}, rightmost={edges[-1]}, n_bins={len(edges) - 1}"
            )

        self.feature_name_to_group = {
            key: f"part_token_id_group_{i}" for i, key in enumerate(bin_edges_dict.keys())
        }
        self.group_to_feature_name = {v: k for k, v in self.feature_name_to_group.items()}

        self.setup_codebooks()

        self.codebook_size = np.prod([len(edges) - 1 for edges in bin_edges_dict.values()])
        self.bin_sizes = [len(edges) - 1 for edges in bin_edges_dict.values()]
        self.list_with_bin_indices = [list(np.arange(0, n_bins)) for n_bins in self.bin_sizes]

        self.list_of_bin_indices_tuples = list(itertools.product(*self.list_with_bin_indices))
        self.list_of_bin_centers = list(
            itertools.product(
                *[self.bin_centers_dict[key] for key in self.bin_centers_dict.keys()]
            )
        )
        self.codebook_size = len(self.list_of_bin_indices_tuples)

        self.lookup_global_token_with_bin_indices = np.zeros(self.bin_sizes)
        # fill the bin_indices_to_global_token array
        for global_token, bin_indices in enumerate(self.list_of_bin_indices_tuples):
            self.lookup_global_token_with_bin_indices[bin_indices] = global_token

    def setup_codebooks(self):
        """Setup the codebooks.

        Each binning feature has its own codebook, which is defined using a torch.nn.Embedding
        layer. This makes it easy to access the continuous values of the bin centers without loops.
        """
        logger.info(100 * "-")
        logger.info("Setting up codebooks...")
        self.codebooks = {}
        for key, bin_centers in self.bin_centers_dict.items():
            # initialize the codebook
            logger.info(100 * "-")
            self.codebooks[key] = nn.Embedding(num_embeddings=len(bin_centers), embedding_dim=1)
            # set the gradient to False
            self.codebooks[key].weight.requires_grad = False
            # set the bin centers as the weights of the codebook
            self.codebooks[key].weight[:, 0] = torch.tensor(bin_centers)
            logger.info(f"Codebook for {key}:  {len(bin_centers)} entries")
            k = 5
            logger.info(f"First {k} entries: {self.codebooks[key].weight[:k, 0]}")
            logger.info(f"Last {k} entries: {self.codebooks[key].weight[-k:, 0]}")

    def get_bin_centers(self, bin_edges):
        """Get the bin centers from the bin edges."""
        return (bin_edges[:-1] + bin_edges[1:]) / 2

    def tokenize_ak_array(
        self,
        ak_arr: ak.Array,
        pp_dict: dict,
        pad_length: int = 128,
    ) -> ak.Array:
        """Tokenize an awkward array.

        Parameters
        ----------
        ak_arr : ak.Array
            The awkward array to tokenize (has to include all the features
            That are specified in the feature dict).
        pp_dict : dict
            Dictionary with preprocessing information.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.

        Returns
        -------
        ak.Array
            The tokenized array (with the global indices/token-ids)
        """
        ak_arr = ak_select_and_preprocess(ak_arr, pp_dict)[:, :pad_length]
        named_bin_indices_ak = ak.Array(
            {
                f"part_token_id_group_{i}": apply_binning(ak_arr[key], edges)
                for i, (key, edges) in enumerate(self.bin_edges_dict.items())
            }
        )
        binned_values_ak = ak.zip(
            {
                key: apply_binning(ak_arr[key], edges, return_binned_values=True)[1]
                for key, edges in self.bin_edges_dict.items()
            }
        )
        # go over the features (keys of the ak array) and bin the values
        # (also extract the bin centers)
        # bin_indices_ak = ak.zip(
        #     [apply_binning(ak_arr[key], edges) for key, edges in self.bin_edges_dict.items()]
        # )
        # convert into global token
        # flatten the bin_indices_ak array to be able to use it as an index
        # bin_indices_ak_flat = ak.to_numpy(ak.flatten(bin_indices_ak))

        # global_indices = ak.values_astype(
        #     [
        #         self.lookup_global_token_with_bin_indices[tuple(idx_tuple)]
        #         for idx_tuple in bin_indices_ak_flat
        #     ],
        #     np.int32,
        # )

        # put global indices back into the awkward array
        # offsets = ak_arr[ak_arr.fields[0]].layout.offsets
        # global_indices = ak.Array(
        #     ak.contents.ListOffsetArray(offsets, ak.Array(global_indices).layout)
        # )

        return ak.Array(
            {
                "part_token_id": named_bin_indices_ak,
                "part_features": ak_arr,
                "part_features_tokenized": binned_values_ak,
            }
            # | ({"bin_indices": bin_indices_ak} if return_bin_indices else {})
            # | ({"named_bin_indices": named_bin_indices_ak} if return_bin_indices else {})
        )

    def reconstruct_ak_tokens(
        self,
        ak_arr: ak.Array,
        pp_dict,
        jets_ak=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        """Reconstruct the original array from the tokenized array using the embedding tables.

        Parameters
        ----------
        ak_arr : ak.Array
            The tokenized array. Will have the field names part_token_id_group_i.
        pp_dict : dict
            Dictionary with preprocessing information.
        jets_ak : ak.Array
            Awkward array of jet-level features, shape (N_jets, N_features_jet).
        pp_dict_jet : dict
            Dictionary with preprocessing information for jet-level features.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.
        """

        present_groups = [key for key in ak_arr.fields if key in self.group_to_feature_name]
        feat_names = [self.group_to_feature_name[key] for key in present_groups]

        max_len = int(ak.max(ak.num(ak_arr[ak_arr.fields[0]])))

        np_arr_reco = np.zeros((len(ak_arr), max_len, len(feat_names)))

        for i, token_group_name in enumerate(present_groups):
            feat_name = self.group_to_feature_name[token_group_name]
            # pad the ak_arr
            tokens = ak_arr[token_group_name]
            tokens_padded, tokens_mask = ak_pad(
                tokens, maxlen=max_len, fill_value=0, return_mask=True
            )
            np_arr_reco[..., i] = F.embedding(
                torch.tensor(ak.to_numpy(tokens_padded)),
                self.codebooks[feat_name].weight,
            )[..., 0].numpy()

        ak_arr_reco_pp = np_to_ak(np_arr_reco, names=feat_names, mask=tokens_mask)
        ak_arr_reco = ak_select_and_preprocess(ak_arr_reco_pp, pp_dict, inverse=True)
        return ak_arr_reco
