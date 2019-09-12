# distutils: language = c++
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
@author: Jordan Graesser
"""

import cython
cimport cython

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector


DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t


cdef extern from 'numpy/npy_math.h' nogil:
    bint npy_isnan(double x)


cdef _create_dictionary_multi_band(unsigned int dict_len, vector[double] values, unsigned int n_bands):

    cdef:
        Py_ssize_t bidx, i
        cpp_map[DTYPE_int64_t, cpp_map[DTYPE_int64_t, vector[double]]] dict_map
        cpp_map[DTYPE_int64_t, vector[double]] dict_map_sub

    with nogil:

        for bidx in range(1, n_bands+1):

            dict_map[bidx] = dict_map_sub

            for i in range(0, dict_len):
                dict_map[bidx][i] = values

    return dict_map


cdef _create_dictionary(unsigned int dict_len, vector[double] values):

    cdef:
        Py_ssize_t i
        cpp_map[DTYPE_int64_t, vector[double]] dict_map

    with nogil:

        for i in range(0, dict_len):
            dict_map[i] = values

    return dict_map


def create_dictionary(unsigned int dict_len,
                      unsigned int n_stats,
                      unsigned int n_bands):

    cdef:
        vector[double] values = np.zeros(n_stats, dtype='float64')

    if n_bands > 0:
        return _create_dictionary_multi_band(dict_len, values, n_bands)
    else:
        return _create_dictionary(dict_len, values)


cdef _merge_dictionary_keys(DTYPE_float64_t[:, ::1] keys_values,
                            DTYPE_int64_t[::1] stat_agg,
                            unsigned int n_stats,
                            unsigned int n_bands):

    cdef:
        Py_ssize_t i, j, jb, bidx
        unsigned int n_features = keys_values.shape[0]
        unsigned int key_value
        cpp_map[DTYPE_int64_t, cpp_map[DTYPE_int64_t, vector[double]]] dict_map
        cpp_map[DTYPE_int64_t, vector[double]] dict_map_sub
        vector[double] values
        vector[double] values_inserted

    with nogil:

        for i in range(0, n_features):

            key_value = <int>keys_values[i, 0]

            for bidx in range(1, n_bands+1):

                if dict_map.find(bidx) == dict_map.end():

                    # Dictionary in the dictionary
                    dict_map[bidx] = dict_map_sub

                # Initiate the key if no key is found
                if dict_map[bidx].find(key_value) == dict_map[bidx].end():

                    # Add the statistics
                    for j in range(1, n_stats+1):
                        values.push_back(keys_values[i, 1 + ((bidx-1) * n_stats) + j - 1])

                    dict_map[bidx][key_value] = values

                else:

                    values_inserted = dict_map[bidx][key_value]

                    for j in range(1, n_stats+1):

                        # Sum
                        if stat_agg[j-1] == 1:
                            values_inserted[1 + ((bidx-1) * n_stats) + j - 1] += keys_values[i, 1 + ((bidx-1) * n_stats) + j - 1]
                        else:
                            values_inserted[1 + ((bidx-1) * n_stats) + j - 1] = ((values_inserted[1 + ((bidx-1) * n_stats) + j - 1] + keys_values[i, 1 + ((bidx-1) * n_stats) + j - 1]) / 2.0)

                    dict_map[bidx][key_value] = values_inserted

                # Reset the list
                for jb in range(0, n_stats):
                    values.pop_back()

    return dict_map


def merge_dictionary_keys(DTYPE_float64_t[:, ::1] keys_values, list stats, unsigned int n_bands):

    cdef:
        Py_ssize_t sidx
        unsigned int n_stats = len(stats)
        DTYPE_int64_t[::1] stat_agg = np.zeros(n_stats, dtype='int64')

    for sidx, stat in enumerate(stats):

        if stat == 'sum':
            stat_agg[sidx] = 1
        else:
            stat_agg[sidx] = 2

    return _merge_dictionary_keys(keys_values, stat_agg, n_stats, n_bands)
