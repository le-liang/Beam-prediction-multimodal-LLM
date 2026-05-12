import numpy as np
import tensorflow as tf

def expand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If ``target_rank`` is smaller than the rank of ``tensor``,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    output = insert_dims(tensor, num_dims, axis)

    return output

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to TensorFlow`s ``expand_dims`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``.
    """
    msg = "`num_dims` must be nonnegative."
    tf.debugging.assert_greater_equal(num_dims, 0, msg)

    rank = tf.rank(tensor)
    msg = "`axis` is out of range `[-(D+1), D]`)"
    tf.debugging.assert_less_equal(axis, rank, msg)
    tf.debugging.assert_greater_equal(axis, -(rank+1), msg)

    axis = axis if axis>=0 else rank+axis+1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis],
                           tf.ones([num_dims], tf.int32),
                           shape[axis:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

def subcarrier_frequencies(num_subcarriers, subcarrier_spacing,
                           dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing


    Input
    ------
    num_subcarriers : int
        Number of subcarriers

    subcarrier_spacing : float
        Subcarrier spacing [Hz]

    dtype : tf.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `tf.complex64` (`tf.float32`).

    Output
    ------
        frequencies : [``num_subcarrier``], tf.float
            Baseband frequencies of subcarriers
    """

    if dtype.is_complex:
        real_dtype = dtype.real_dtype
    elif dtype.if_floating:
        real_dtype = dtype
    else:
        raise AssertionError("dtype must be a complex or floating datatype")

    if tf.equal(tf.math.floormod(num_subcarriers, 2), 0):
        start=-num_subcarriers/2
        limit=num_subcarriers/2
    else:
        start=-(num_subcarriers-1)/2
        limit=(num_subcarriers-1)/2+1

    frequencies = tf.range( start=start,
                            limit=limit,
                            dtype=real_dtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    """
    Convert CIR to ofdm channel, credit to Sionna

    Parameters
    ----------
    batch : dict

    Returns
    -------
    processed_batch : dict
        Updated lidar batch.
    """
    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    h = tf.expand_dims(a, axis=-1)
    frequencies = expand_to_rank(frequencies, tf.rank(tau), axis=0)

    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = tf.exp(tf.complex(tf.constant(0, real_dtype),
        -2*np.pi*frequencies*tau))
    h_f = h*e
    # Sum over all clusters to get the channel frequency responses
    h_f = tf.reduce_sum(h_f, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = tf.reduce_mean(tf.square(tf.abs(h_f)), axis=(2,4,5,6),
                            keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        h_f = tf.math.divide_no_nan(h_f, c)

    return h_f

def DFT_codebook_3D(bs_ant_shape, bs_beam_shape):
    num_ant_v, num_ant_h = bs_ant_shape
    num_beam_v, num_beam_h = bs_beam_shape
    num_ant_total = num_ant_v * num_ant_h
    num_beam_total = num_beam_v * num_beam_h

    idx_ant_h = np.arange(num_ant_h).reshape(num_ant_h, 1)

    angles_h = -np.pi + (2 * np.pi / num_beam_h) * np.arange(0.5, num_beam_h + 0.5, 1)

    beam_vectors_h = np.exp(-1j * idx_ant_h * (angles_h.reshape(1, num_beam_h))) / np.sqrt(num_ant_h)

    candidate_narrow_beam = np.zeros((num_ant_total, num_beam_total), dtype=complex)

    beam_idx_flat = 0
    for b_h_idx in range(num_beam_h):
        candidate_narrow_beam[:, beam_idx_flat] = beam_vectors_h[:, b_h_idx]
        beam_idx_flat += 1
    return candidate_narrow_beam


def channel_eff(H, H_rx_ant_shape, H_tx_ant_shape, rx_ant_shape, tx_ant_shape):

    def center_indices(array_shape, sub_shape):
        row_start = array_shape[0] // 2 - sub_shape[0] // 2
        col_start = array_shape[1] // 2 - sub_shape[1] // 2
        rows = np.arange(row_start, row_start + sub_shape[0])
        cols = np.arange(col_start, col_start + sub_shape[1])
        grid = np.array([c * array_shape[0] + r for c in cols for r in rows])
        return grid

    tx_idx = center_indices(H_tx_ant_shape, tx_ant_shape)  # 16 发射天线索引
    rx_idx = center_indices(H_rx_ant_shape, rx_ant_shape)  # 16 接收天线索引

    H_sub = H[np.ix_(rx_idx, tx_idx)]  # shape: [16, 16, N_f]
    return H_sub
