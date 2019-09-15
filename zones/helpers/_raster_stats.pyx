# distutils: language = c++
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc

from mpglue import raster_tools, vector_tools


DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t


cdef _process_chunk(DTYPE_float64_t[:, :, ::-1] vchunk,
                    DTYPE_float64_t[:, :, ::-1] zchunk,
                    unsigned int n_bands,
                    unsigned int n_rows,
                    unsigned int n_cols,
                    cpp_map[DTYPE_int64_t, cpp_map[DTYPE_int64_t, vector[double]]] dict_map):

    cdef:
        Py_ssize_t m, n
        double band_value, zone_value
        unsigned int feature_key
        vector[double] values

    # TODO: parallel over bands
    with nogil:

        for b in range(0, n_bands):

            for m in range(0, n_rows):

                for n in range(0, n_cols):

                    band_value = vchunk[b, m, n]
                    zone_value = zchunk[m, n]
                    feature_key = <int>zone_value

                    # Get the zone data list if it exists
                    if dict_map[b+1].find(feature_key) != dict_map[b+1].end():
                        values = dict_map[b+1][feature_key]

                    # Add the current value
                    values.push_back(band_value)

                    # Update the feature for the current band
                    dict_map[b+1][feature_key] = values

    return dict_map


cdef _calc_dict_sub(cpp_map[DTYPE_int64_t, vector[double]] dict_map_sub) nogil:

    cdef:
        vector[double] values

    # TODO: parallel over features
    with nogil:

        for it in dict_map_sub:

            values = it.second

            vit = values.begin()

            while vit != values.end():
                deref(vit)


cdef _calc_stats(cpp_map[DTYPE_int64_t, cpp_map[DTYPE_int64_t, vector[double]]] dict_map,
                 unsigned int n_bands):

    cdef:
        Py_ssize_t b

    for b in range(0, n_bands):
        dict_map[b+1] = _calc_dict_sub(dict_map[b+1])





def calc_stats(str values, str zones, int chunk_size=1000):

    cdef:
        Py_ssize_t i, j, bidx
        float min_top, max_bottom, max_left, min_right
        float cell_size
        float vx, vy, zx, zy
        unsigned int rows, cols, n_rows, n_cols, n_bands
        DTYPE_float64_t[:, :, ::-1] vchunk, zchunk
        cpp_map[DTYPE_int64_t, cpp_map[DTYPE_int64_t, vector[double]]] dict_map
        cpp_map[DTYPE_int64_t, vector[double]] dict_map_sub

    vsrc = raster_tools.ropen(values)
    zsrc = raster_tools.ropen(zones)

    cell_size = vsrc.cellY
    n_bands = vsrc.bands

    # Get the starting row coordinate
    min_top = min(vsrc.top, zsrc.top)

    # Get the ending row coordinate
    max_bottom = max(vsrc.bottom, zsrc.bottom)

    # Get the starting column coordinate
    max_left = max(vsrc.left, zsrc.left)

    # Get the ending column coordinate
    min_right = min(vsrc.right, zsrc.right)

    vx, vy = vector_tools.get_xy_offsets(image_list=[vsrc.left, vsrc.top, vsrc.right, vsrc.bottom, cell_size, cell_size],
                                         x=max_left,
                                         y=min_top)[2:]

    zx, zy = vector_tools.get_xy_offsets(image_list=[zsrc.left, zsrc.top, zsrc.right, zsrc.bottom, cell_size, cell_size],
                                         x=max_left,
                                         y=min_top)[2:]

    rows = <int>((min_top - max_bottom) / cell_size)
    cols = <int>((min_right - max_left) / cell_size)

    # Prepare the dictionary
    with gil:

        for bidx in range(1, n_bands+1):
            dict_map[bidx] = dict_map_sub

    # Iterate over the images in chunks
    for i from 0 <= i < rows by chunk_size:

        n_rows = raster_tools.n_rows_cols(i, chunk_size, rows)

        for j from 0 <= j < cols by chunk_size:

            n_cols = raster_tools.n_rows_cols(j, chunk_size, cols)

            vchunk = vsrc.read(i=i+vy, j=j+vx, rows=n_rows, cols=n_cols, bands=-1)
            zchunk = zsrc.read(i=i+zy, j=j+zx, rows=n_rows, cols=n_cols, bands=-1)

            # Process the chunk, iterating over layers in parallel
            dict_map = _process_chunk(vchunk, zchunk, n_rows, n_cols, dict_map)

    # Calculate the statistics
    _calc_stats(dict_map)

