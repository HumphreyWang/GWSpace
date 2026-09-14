#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 En-Kun Li, Han Wang
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate space detector response for multiple TDI generations in both time domain and frequency domain.
 Support 'XYZAET' channel currently. For t-domain, it will return the responsed waveform,
 while for f-domain, it will return the transfer function,
 and one should multiply it by the original waveform manually."""

import numpy as np
from numba import njit
from gwspace.Orbit import detectors


@njit(inline='always')
def _matrix_res_pro_scalar(n0, n1, n2, p):
    """TensorProduct(n, n):p for one frequency point."""
    return (n0*p[0, 0]*n0 + n0*p[0, 1]*n1 + n0*p[0, 2]*n2
            + n1*p[1, 0]*n0 + n1*p[1, 1]*n1 + n1*p[1, 2]*n2
            + n2*p[2, 0]*n0 + n2*p[2, 1]*n1 + n2*p[2, 2]*n2)


@njit
def _matrix_res_pro(n, p):
    """TensorProduct(n, n) : P,  where A:B = A_ij B_ij"""
    response = np.empty(n.shape[1], dtype=p.dtype)
    for i in range(n.shape[1]):
        response[i] = _matrix_res_pro_scalar(n[0, i], n[1, i], n[2, i], p)
    return response


def get_y_slr_td(wf, tf, det, TDIgen=1):
    """TODO: here we calculate orbits of the detectors only once at tf, but considering TDI_delay here,
         their positions need to be **recalculated**, e.g. at tf-L, tf-2*L, ..."""
    if TDIgen == 1:
        TDI_delay = 4
    elif TDIgen == 2:
        TDI_delay = 8
    else:
        raise NotImplementedError

    det = detectors[det](tf)
    p1, p2, p3 = det.orbits
    L = det.L_T
    n1 = det.uni_vec_ij(3, 2)
    n2 = det.uni_vec_ij(1, 3)
    n3 = det.uni_vec_ij(2, 1)

    k = wf.vec_k
    p_p, p_c = wf.polarization()
    xi1 = (_matrix_res_pro(n1, p_p), _matrix_res_pro(n1, p_c))
    xi2 = (_matrix_res_pro(n2, p_p), _matrix_res_pro(n2, p_c))
    xi3 = (_matrix_res_pro(n3, p_p), _matrix_res_pro(n3, p_c))

    tf_kp1 = tf - np.dot(k, p1)
    tf_kp2 = tf - np.dot(k, p2)
    tf_kp3 = tf - np.dot(k, p3)
    kn1 = np.dot(k, n1)
    kn2 = np.dot(k, n2)
    kn3 = np.dot(k, n3)
    del p1, p2, p3, n1, n2, n3, det

    # Here `i` is for `i*L` TDI_delay, in 1st generation TDI we consider delay up to 4,
    # i.e. t-0L, t-1L, t-2L, t-3L, t-4L. And then we calculate difference between each L delay.
    # Evaluate the waveform once per spacecraft and delay; each value is reused by its two arms.
    h_p1 = [wf.get_hphc(tf_kp1 - i_*L) for i_ in range(TDI_delay+1)]
    h_p2 = [wf.get_hphc(tf_kp2 - i_*L) for i_ in range(TDI_delay+1)]
    h_p3 = [wf.get_hphc(tf_kp3 - i_*L) for i_ in range(TDI_delay+1)]
    del tf_kp1, tf_kp2, tf_kp3

    def get_y(h_sender, h_receiver, xi, denominator):
        xi_p, xi_c = xi
        return [((h_sender[i+1][0]*xi_p + h_sender[i+1][1]*xi_c)
                 - (h_receiver[i][0]*xi_p + h_receiver[i][1]*xi_c))/denominator
                for i in range(TDI_delay)]

    y_slr = {(1, 2): get_y(h_p1, h_p2, xi3, 2*(1+kn3)),
             (2, 1): get_y(h_p2, h_p1, xi3, 2*(1-kn3)),
             (1, 3): get_y(h_p1, h_p3, xi2, 2*(1-kn2)),
             (3, 1): get_y(h_p3, h_p1, xi2, 2*(1+kn2)),
             (2, 3): get_y(h_p2, h_p3, xi1, 2*(1+kn1)),
             (3, 2): get_y(h_p3, h_p2, xi1, 2*(1-kn1))}
    return y_slr


@njit(inline='always')
def tdi_XYZ2AET(X, Y, Z):
    """Calculate the orthogonal AET channels from the TDI XYZ channels.

    This is the common XYZ-to-AET definition used by both the time-domain
    response and the frequency-domain Numba kernel.
    """
    A = 1/np.sqrt(2)*(Z-X)
    E = 1/np.sqrt(6)*(X-2*Y+Z)
    T = 1/np.sqrt(3)*(X+Y+Z)
    return A, E, T


def get_XYZ_td(wf, tf, det='TQ', TDIgen=1):
    """ Generate TDI XYZ in different TDI generation """
    y_slr = get_y_slr_td(wf, tf, det, TDIgen)
    y31, y13 = y_slr[(3, 1)], y_slr[(1, 3)]
    y12, y21 = y_slr[(1, 2)], y_slr[(2, 1)]
    y23, y32 = y_slr[(2, 3)], y_slr[(3, 2)]

    if TDIgen == 1:
        X = (y31[0]+y13[1]+y21[2]+y12[3] - y21[0]-y12[1]-y31[2]-y13[3])
        Y = (y12[0]+y21[1]+y32[2]+y23[3] - y32[0]-y23[1]-y12[2]-y21[3])
        Z = (y23[0]+y32[1]+y13[2]+y31[3] - y13[0]-y31[1]-y23[2]-y32[3])
    elif TDIgen == 2:
        X = (y31[0]+y13[1]+y21[2]+y12[3] + y21[4]+y12[5]+y31[6]+y13[7]
             - y21[0]-y12[1]-y31[2]-y13[3] - y31[4]-y13[5]-y21[6]-y12[7])
        Y = (y12[0]+y21[1]+y32[2]+y23[3] + y32[4]+y23[5]+y12[6]+y21[7]
             - y32[0]-y23[1]-y12[2]-y21[3] - y12[4]-y21[5]-y32[6]-y23[7])
        Z = (y23[0]+y32[1]+y13[2]+y31[3] + y13[4]+y31[5]+y23[6]+y32[7]
             - y13[0]-y31[1]-y23[2]-y32[3] - y23[4]-y32[5]-y13[6]-y31[7])
    else:
        raise NotImplementedError

    return X, Y, Z


def get_AET_td(wf, tf, det='TQ', TDIgen=1):
    """ Generate TDI AET in different TDI generation """
    X, Y, Z = get_XYZ_td(wf, tf, det, TDIgen)
    A, E, T = tdi_XYZ2AET(X, Y, Z)
    return A, E, T


_Y_SLR = 0
_XYZ = 1
_AET = 2

_LINKS = ((1, 2), (2, 1),
          (1, 3), (3, 1),
          (2, 3), (3, 2))


@njit(inline='always')
def _sinc_scalar(x):
    """Equivalent to np.sinc(x) for a scalar."""
    if x == 0.:
        return 1.
    return np.sin(np.pi*x)/(np.pi*x)


@njit(inline="always")
def _link_pair_fd(f, L, k_dot_u, k_dot_positions):
    """Calculate a pair of single-link responses.

    See Marsat et al. (Eqs. 21 and 28):
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011
    """
    phase = np.pi * f * (L + k_dot_positions)

    # com_f = 1j/2*pi*f*L in Marsat et al. because their A, E, T are 1/2
    # of the LDC definitions. See Eq. 2 of McWilliams et al.:
    # https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.064014
    common = 1j * np.pi * f * L * (np.cos(phase) + 1j * np.sin(phase))

    # np.sinc(x) is defined as sin(pi*x)/(pi*x).
    forward = common * _sinc_scalar(f * L * (1.0 - k_dot_u))
    backward = common * _sinc_scalar(f * L * (1.0 + k_dot_u))
    return forward, backward


@njit(inline="always")
def _xyz_from_links(y12, y21, y13, y31, y23, y32, delay, factor):
    """Combine six single-link responses into equal-arm Michelson XYZ."""
    X = factor * (y31 + delay * y13 - y21 - delay * y12)
    Y = factor * (y12 + delay * y21 - y32 - delay * y23)
    Z = factor * (y23 + delay * y32 - y13 - delay * y31)
    return X, Y, Z


@njit
def _trans_fd_kernel(vec_k, tensors, p1, p2, p3, f, L, TDIgen, output):
    n_samples = max(len(f), p1.shape[1])
    n_channels = 6 if output == _Y_SLR else 3
    response = np.empty((len(tensors), n_channels, n_samples), dtype=np.complex128)

    for i in range(n_samples):
        fi = f[0] if len(f) == 1 else f[i]
        orbit_i = 0 if p1.shape[1] == 1 else i

        u12_0 = (p2[0, orbit_i] - p1[0, orbit_i]) / L
        u12_1 = (p2[1, orbit_i] - p1[1, orbit_i]) / L
        u12_2 = (p2[2, orbit_i] - p1[2, orbit_i]) / L

        u23_0 = (p3[0, orbit_i] - p2[0, orbit_i]) / L
        u23_1 = (p3[1, orbit_i] - p2[1, orbit_i]) / L
        u23_2 = (p3[2, orbit_i] - p2[2, orbit_i]) / L

        u13_0 = (p3[0, orbit_i] - p1[0, orbit_i]) / L
        u13_1 = (p3[1, orbit_i] - p1[1, orbit_i]) / L
        u13_2 = (p3[2, orbit_i] - p1[2, orbit_i]) / L

        k_dot_u12 = vec_k[0] * u12_0 + vec_k[1] * u12_1 + vec_k[2] * u12_2
        k_dot_u23 = vec_k[0] * u23_0 + vec_k[1] * u23_1 + vec_k[2] * u23_2
        k_dot_u13 = vec_k[0] * u13_0 + vec_k[1] * u13_1 + vec_k[2] * u13_2

        k_dot_p12 = (vec_k[0] * (p1[0, orbit_i] + p2[0, orbit_i])
                     + vec_k[1] * (p1[1, orbit_i] + p2[1, orbit_i])
                     + vec_k[2] * (p1[2, orbit_i] + p2[2, orbit_i]))
        k_dot_p23 = (vec_k[0] * (p2[0, orbit_i] + p3[0, orbit_i])
                     + vec_k[1] * (p2[1, orbit_i] + p3[1, orbit_i])
                     + vec_k[2] * (p2[2, orbit_i] + p3[2, orbit_i]))
        k_dot_p31 = (vec_k[0] * (p3[0, orbit_i] + p1[0, orbit_i])
                     + vec_k[1] * (p3[1, orbit_i] + p1[1, orbit_i])
                     + vec_k[2] * (p3[2, orbit_i] + p1[2, orbit_i]))

        y12_pre, y21_pre = _link_pair_fd(fi, L, k_dot_u12, k_dot_p12)
        y23_pre, y32_pre = _link_pair_fd(fi, L, k_dot_u23, k_dot_p23)
        y13_pre, y31_pre = _link_pair_fd(fi, L, k_dot_u13, k_dot_p31)

        delay = 0j
        factor = 0j

        if output != _Y_SLR:
            # Time delay factor Dt = exp(2j*pi*f*L).
            delay_phase = 2.0 * np.pi * fi * L
            delay = np.cos(delay_phase) + 1j * np.sin(delay_phase)
            delay2 = delay * delay

            factor = 1.0 - delay2
            if TDIgen == 2:
                factor *= 1.0 - delay2 * delay2

        for j in range(len(tensors)):
            tensor = tensors[j]

            n12pn12 = _matrix_res_pro_scalar(u12_0, u12_1, u12_2, tensor)
            n23pn23 = _matrix_res_pro_scalar(u23_0, u23_1, u23_2, tensor)
            n13pn13 = _matrix_res_pro_scalar(u13_0, u13_1, u13_2, tensor)

            y12 = y12_pre * n12pn12
            y21 = y21_pre * n12pn12
            y13 = y13_pre * n13pn13
            y31 = y31_pre * n13pn13
            y23 = y23_pre * n23pn23
            y32 = y32_pre * n23pn23

            if output == _Y_SLR:
                response[j, 0, i] = y12
                response[j, 1, i] = y21
                response[j, 2, i] = y13
                response[j, 3, i] = y31
                response[j, 4, i] = y23
                response[j, 5, i] = y32
                continue

            X, Y, Z = _xyz_from_links(y12, y21, y13, y31, y23, y32, delay, factor)

            if output == _XYZ:
                response[j, 0, i] = X
                response[j, 1, i] = Y
                response[j, 2, i] = Z
            else:
                A, E, T = tdi_XYZ2AET(X, Y, Z)
                response[j, 0, i] = A
                response[j, 1, i] = E
                response[j, 2, i] = T

    return response


def _trans_fd(vec_k, p, det, f, TDIgen, output):
    if TDIgen not in (1, 2):
        raise NotImplementedError

    if type(p) is not tuple:
        p = (p,)

    f = np.atleast_1d(f)
    p1, p2, p3 = (np.asarray(position).reshape(3, -1) for position in det.orbits)
    if len(f) != 1 and p1.shape[1] != 1 and len(f) != p1.shape[1]:
        raise ValueError("f and detector orbits must have equal lengths, unless one has length 1")

    return _trans_fd_kernel(vec_k, np.asarray(p), p1, p2, p3, f, det.L_T, TDIgen, output)


def trans_y_slr_fd(vec_k, p, det, f):
    """Calculate the six single-link frequency-domain responses.

    See Marsat et al. (Eqs. 21 and 28):
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011
    """
    response = _trans_fd(vec_k, p, det, f, TDIgen=1, output=_Y_SLR)

    return tuple({link: tensor_response[i] for i, link in enumerate(_LINKS)}
                 for tensor_response in response)


def trans_XYZ_fd(vec_k, p, det, f, TDIgen=1):
    """ Calculate XYZ from y_slr in frequency domain, unlike in time domain, it returns the transfer function.
     To get the responsed waveform, you need to multiply it by the original waveform manually.

    :param vec_k: the prop direction of GWs (x,y,z), which is determined by (lambda, beta)
    :param p: p is a tuple, which contains P_lm (or P_x & P_+, i.e. e^+ & e^x in $h = h_+ e^+ + h_x e^x$)
    :param det: GW detector Orbit object
    :param f: frequency scalar or array
    :param TDIgen: TDI generation
    :return: tuple with same length of tuple p
    """
    response = _trans_fd(vec_k, p, det, f, TDIgen, output=_XYZ)
    return tuple(response)


def trans_AET_fd(vec_k, p, det, f, TDIgen=1):
    """ Calculate AET from y_slr in frequency domain, unlike in time domain, it returns the transfer function.
     To get the responsed waveform, you need to multiply it by the original waveform manually.

    :param vec_k: the prop direction of GWs (x,y,z), which is determined by (lambda, beta)
    :param p: p is a tuple, which contains P_lm (or P_x & P_+, i.e. e^+ & e^x in $h = h_+ e^+ + h_x e^x$)
    :param det: GW detector Orbit object
    :param f: frequency scalar or array
    :param TDIgen: TDI generation
    :return: tuple with same length of tuple p
    """
    response = _trans_fd(vec_k, p, det, f, TDIgen, output=_AET)
    return tuple(response)
