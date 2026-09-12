#!/usr/bin/env python

"""Time-domain TDI responses for linearly parametrized ringdown modes."""

import numpy as np

from gwspace.Orbit import detectors
from gwspace.response import _matrix_res_pro, tdi_XYZ2AET


def get_y_slr_basis_td(wf, tf, modes=None, det='TQ', TDIgen=1):
    """Build the two linear ringdown bases for every directed link.

    Following Eqs. (A.5)-(A.12) of arXiv:2604.20914,

        y_slr(t) = B1 * g1_slr(t) + B2 * g2_slr(t)

    where (B1, B2) are linear amplitude parameters per mode.

    ``wf.get_base_func(times, mode)`` returns the damped cosine and sine
    functions C_lmn and S_lmn. It must accept a two-dimensional time array
    and preserve its shape. ``wf.modes_dic[mode]`` supplies the corresponding
    angular factors Y^+_lm and Y^x_lm.

    The present implementation uses one constant arm light-travel time and
    evaluates the detector geometry at ``tf`` rather than at every retarded
    time. Thus its TDI-2 path is the equal-arm, frozen-geometry form, not a
    general flexing unequal-arm second-generation TDI calculation.
    """
    if TDIgen == 1:
        TDI_delay = 4
    elif TDIgen == 2:
        TDI_delay = 8
    else:
        raise NotImplementedError
    if modes is None:
        modes = wf.modes_list

    tf = np.asarray(tf)
    if tf.ndim != 1:
        raise ValueError("tf must be a one-dimensional time array")
    det_obj = detectors[det](tf)
    p1, p2, p3 = det_obj.orbits
    L = det_obj.L_T
    n1 = det_obj.uni_vec_ij(3, 2)
    n2 = det_obj.uni_vec_ij(1, 3)
    n3 = det_obj.uni_vec_ij(2, 1)

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
    del det_obj, p1, p2, p3, n1, n2, n3

    # p1, p2, p3 and L_T are measured in seconds, so tf-k.r and i*L are
    # respectively the propagation delay and TDI delay in Eqs. (A.8)-(A.11).
    # Flatten (spacecraft, delay, time) to one 2-D batch for get_base_func,
    # then restore those three axes with reshape below.
    delay_stack = [tf_kp - i*L for tf_kp in (tf_kp1, tf_kp2, tf_kp3) for i in range(TDI_delay+1)]
    flat_shape = (len(delay_stack), len(tf))
    basis_shape = (3, TDI_delay+1, len(tf))
    del tf_kp1, tf_kp2, tf_kp3

    # n1=n_32, n2=n_13, n3=n_21. Equation (A.11) uses the photon direction
    # n_sr and denominator 2*(1-k.n_sr), which gives the signs below.
    link_table = {(1, 2): dict(zeta=xi3, denom=2*(1+kn3)),
                  (2, 1): dict(zeta=xi3, denom=2*(1-kn3)),
                  (1, 3): dict(zeta=xi2, denom=2*(1-kn2)),
                  (3, 1): dict(zeta=xi2, denom=2*(1+kn2)),
                  (2, 3): dict(zeta=xi1, denom=2*(1+kn1)),
                  (3, 2): dict(zeta=xi1, denom=2*(1-kn1)), }

    y_basis_modes = []
    for mode in modes:
        mode_meta = wf.modes_dic[mode]
        y_plus = mode_meta["Y_lm_p"]
        y_cross = mode_meta["Y_lm_m"]

        c, s = wf.get_base_func(delay_stack, mode)
        c, s = np.asarray(c), np.asarray(s)
        if c.shape != flat_shape or s.shape != flat_shape:
            raise ValueError("get_base_func must preserve the shape of its time input")
        c = c.reshape(basis_shape)
        s = s.reshape(basis_shape)

        y_basis_one = {}
        for key, meta in link_table.items():
            zeta_p, zeta_x = meta["zeta"]
            denom = meta["denom"]
            cs, ss = c[key[0] - 1], s[key[0] - 1]  # sender
            cr, sr = c[key[1] - 1], s[key[1] - 1]  # receiver

            # For link s->r at the additional TDI delay i*L,
            #   dC = C(t-(i+1)L-k.r_s) - C(t-iL-k.r_r),
            #   dS = S(t-(i+1)L-k.r_s) - S(t-iL-k.r_r).
            # These are the bracketed differences in Eq. (A.11).
            dc = cs[1:] - cr[:-1]
            ds = ss[1:] - sr[:-1]

            # Equation (A.11): g1 multiplies B_lmn,1=A_lmn*cos(phi_lmn),
            # while g2 multiplies B_lmn,2=A_lmn*sin(phi_lmn).
            g1 = (zeta_p*y_plus*dc + zeta_x*y_cross*ds) / denom
            g2 = (-zeta_p*y_plus*ds + zeta_x*y_cross*dc) / denom
            y_basis_one[key] = (g1, g2)

        y_basis_modes.append(y_basis_one)

    return y_basis_modes


def get_XYZ_basis_td(wf, tf, modes=None, det='TQ', TDIgen=1):
    """Assemble the XYZ bases by applying the TDI delays to each link basis.

    This implements the linear construction in Eqs. (A.14)-(A.15) of
    arXiv:2604.20914, e.g. X(t) = B1*G_X1(t) + B2*G_X2(t).
    """
    y_basis = get_y_slr_basis_td(wf, tf, modes, det, TDIgen)

    channel_basis_modes = []
    for yb in y_basis:
        (g31_1, g31_2), (g13_1, g13_2) = yb[(3, 1)], yb[(1, 3)]
        (g12_1, g12_2), (g21_1, g21_2) = yb[(1, 2)], yb[(2, 1)]
        (g23_1, g23_2), (g32_1, g32_2) = yb[(2, 3)], yb[(3, 2)]

        # Each g array is indexed by an additional integer delay i*L. The
        # combinations below apply the delay operators of Eqs. (A.13)-(A.15).
        if TDIgen == 1:
            GX1 = (g31_1[0]+g13_1[1]+g21_1[2]+g12_1[3]
                   - g21_1[0]-g12_1[1]-g31_1[2]-g13_1[3])
            GX2 = (g31_2[0]+g13_2[1]+g21_2[2]+g12_2[3]
                   - g21_2[0]-g12_2[1]-g31_2[2]-g13_2[3])
            GY1 = (g12_1[0]+g21_1[1]+g32_1[2]+g23_1[3]
                   - g32_1[0]-g23_1[1]-g12_1[2]-g21_1[3])
            GY2 = (g12_2[0]+g21_2[1]+g32_2[2]+g23_2[3]
                   - g32_2[0]-g23_2[1]-g12_2[2]-g21_2[3])
            GZ1 = (g23_1[0]+g32_1[1]+g13_1[2]+g31_1[3]
                   - g13_1[0]-g31_1[1]-g23_1[2]-g32_1[3])
            GZ2 = (g23_2[0]+g32_2[1]+g13_2[2]+g31_2[3]
                   - g13_2[0]-g31_2[1]-g23_2[2]-g32_2[3])
        elif TDIgen == 2:
            GX1 = (g31_1[0]+g13_1[1]+g21_1[2]+g12_1[3] + g21_1[4]+g12_1[5]+g31_1[6]+g13_1[7]
                   - g21_1[0]-g12_1[1]-g31_1[2]-g13_1[3] - g31_1[4]-g13_1[5]-g21_1[6]-g12_1[7])
            GX2 = (g31_2[0]+g13_2[1]+g21_2[2]+g12_2[3] + g21_2[4]+g12_2[5]+g31_2[6]+g13_2[7]
                   - g21_2[0]-g12_2[1]-g31_2[2]-g13_2[3] - g31_2[4]-g13_2[5]-g21_2[6]-g12_2[7])
            GY1 = (g12_1[0]+g21_1[1]+g32_1[2]+g23_1[3] + g32_1[4]+g23_1[5]+g12_1[6]+g21_1[7]
                   - g32_1[0]-g23_1[1]-g12_1[2]-g21_1[3] - g12_1[4]-g21_1[5]-g32_1[6]-g23_1[7])
            GY2 = (g12_2[0]+g21_2[1]+g32_2[2]+g23_2[3] + g32_2[4]+g23_2[5]+g12_2[6]+g21_2[7]
                   - g32_2[0]-g23_2[1]-g12_2[2]-g21_2[3] - g12_2[4]-g21_2[5]-g32_2[6]-g23_2[7])
            GZ1 = (g23_1[0]+g32_1[1]+g13_1[2]+g31_1[3] + g13_1[4]+g31_1[5]+g23_1[6]+g32_1[7]
                   - g13_1[0]-g31_1[1]-g23_1[2]-g32_1[3] - g23_1[4]-g32_1[5]-g13_1[6]-g31_1[7])
            GZ2 = (g23_2[0]+g32_2[1]+g13_2[2]+g31_2[3] + g13_2[4]+g31_2[5]+g23_2[6]+g32_2[7]
                   - g13_2[0]-g31_2[1]-g23_2[2]-g32_2[3] - g23_2[4]-g32_2[5]-g13_2[6]-g31_2[7])
        else:
            raise NotImplementedError

        channel_basis_modes.append({'X': (GX1, GX2), 'Y': (GY1, GY2), 'Z': (GZ1, GZ2)})

    return channel_basis_modes


def get_AET_basis_td(wf, tf, modes=None, det='TQ', TDIgen=1):
    """Convert each mode's XYZ linear bases to the orthogonal AET channels."""
    xyz_basis_modes = get_XYZ_basis_td(wf, tf, modes, det, TDIgen)

    channel_basis_modes = []
    for xyz_basis in xyz_basis_modes:
        X1, X2 = xyz_basis['X']
        Y1, Y2 = xyz_basis['Y']
        Z1, Z2 = xyz_basis['Z']
        A1, E1, T1 = tdi_XYZ2AET(X1, Y1, Z1)
        A2, E2, T2 = tdi_XYZ2AET(X2, Y2, Z2)
        channel_basis_modes.append({'A': (A1, A2),
                                    'E': (E1, E2),
                                    'T': (T1, T2)})

    return channel_basis_modes
