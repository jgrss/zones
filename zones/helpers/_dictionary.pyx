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


cdef _merge_dictionary_keys(DTYPE_float64_t[:, ::1] keys_values):

    cdef:
        Py_ssize_t i
        unsigned int n_features = keys_values.shape[0]
        cpp_map[DTYPE_int64_t, double] dict_map

    with nogil:

        for i in range(0, n_features):
            dict_map[<int>keys_values[i, 0]] += keys_values[i, 1]

    return dict_map


def merge_dictionary_keys(DTYPE_float64_t[:, ::1] keys_values):
    return _merge_dictionary_keys(keys_values)
