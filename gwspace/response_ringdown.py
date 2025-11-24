#!/usr/bin/env python

import numpy as np
from numba import njit
from gwspace.Orbit import detectors
from gwspace.response import _matrix_res_pro


def get_y_slr_basis_td(wf, tf, modes=None, det='TQ', TDIgen=1):
    """ Build link-level basis for j=1,2 so that:
        y_slr(t) = B1 * g1_slr(t) + B2 * g2_slr(t)
    where (B1, B2) are linear amplitude parameters per mode.
    """
    if TDIgen == 1:
        TDI_delay = 4
    elif TDIgen == 2:
        TDI_delay = 8
    else:
        raise NotImplementedError
    if modes is None:
        modes = wf.modes_list()

    tf = np.array(tf)
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

    stack1 = [tf_kp1 - i * L for i in range(TDI_delay+1)]
    stack2 = [tf_kp2 - i * L for i in range(TDI_delay+1)]
    stack3 = [tf_kp3 - i * L for i in range(TDI_delay+1)]
    link_table = {(1, 2): dict(zeta=xi3, denom=2*(1+kn3)),
                  (2, 1): dict(zeta=xi3, denom=2*(1-kn3)),
                  (1, 3): dict(zeta=xi2, denom=2*(1-kn2)),
                  (3, 1): dict(zeta=xi2, denom=2*(1+kn2)),
                  (2, 3): dict(zeta=xi1, denom=2*(1+kn1)),
                  (3, 2): dict(zeta=xi1, denom=2*(1-kn1)), }

    y_basis_modes = []
    for mode in modes:
        c = [wf.c_func(stack, mode) for stack in [stack1, stack2, stack3]]
        s = [wf.s_func(stack, mode) for stack in [stack1, stack2, stack3]]
        # or wf.c_func(stack, mode) -> [wf.c_func(t, mode) for t in stack]

        y_basis_one = {}
        for key, meta in link_table.items():
            zeta_p, zeta_x = meta["zeta"]
            denom = meta["denom"]
            cs, ss = c[key[0] - 1], s[key[0] - 1]  # sender
            cr, sr = c[key[1] - 1], s[key[1] - 1]  # receiver

            g1, g2 = [], []
            for i in range(TDI_delay):
                # Δc = c(t - L - k·r_s) - c(t - k·r_r)
                dc = cs[i + 1] - cr[i]
                ds = ss[i + 1] - sr[i]
                # g1 = [ ζ^+ Δc + ζ^× Δs ] / denom
                # g2 = [ -ζ^+ Δs + ζ^× Δc ] / denom
                g1.append((zeta_p * dc + zeta_x * ds) / denom)
                g2.append((-zeta_p*ds+zeta_x*dc)/denom)

            g1 = np.stack(g1, axis=0)
            g2 = np.stack(g2, axis=0)
            y_basis_one[key] = (g1, g2)

        y_basis_modes.append(y_basis_one)

    return y_basis_modes


def get_XYZ_basis_td(wf, tf, modes=None, det='TQ', TDIgen=1):
    """ Assemble TDI {X,Y,Z} basis so that X(t) = B1 * G_X1(t) + B2 * G_X2(t),  etc. """
    y_basis = get_y_slr_basis_td(wf, tf, modes, det, TDIgen)

    channel_basis_modes = []
    for yb in y_basis:
        (g31_1, g31_2), (g13_1, g13_2) = yb[(3, 1)], yb[(1, 3)]
        (g12_1, g12_2), (g21_1, g21_2) = yb[(1, 2)], yb[(2, 1)]
        (g23_1, g23_2), (g32_1, g32_2) = yb[(2, 3)], yb[(3, 2)]

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


def build_firefly_basis(self, gw_params, modes=None, network=None, channels=None, TDIgen=1):
    """
    构造 Firefly 线性参数的基底波形。

    返回
    ----
    basis_dict[det][ch] = [h_0, h_1, ...]  # 下标与 param_names 对应
    param_names : list[str]                # 如 ["Bc220","Bs220",...]
    """
    # network / channels，风格与 network_inner_product 一致
    if network is None:
        network = self.det_list
    else:
        network = list(network)

    if channels is None:
        channels = self.channels
    else:
        channels = tuple(channels)

    # 展开 lmns，确定模式列表
    lmn_all = self._expand_lmns(gw_params)
    if modes is None:
        modes_use = lmn_all
    else:
        modes_use = list(modes)
        for lmn in modes_use:
            if lmn not in lmn_all:
                raise ValueError(
                    f"模式 {lmn} 不在 gw_params['lmns'] 展开列表 {lmn_all}"
                )

    # 基底母参数：所有模式 amp/phi 先清零
    base_gw_params = gw_params.copy()
    for lmn in lmn_all:
        base_gw_params[f"amp{lmn}"] = 0.0
        base_gw_params[f"phi{lmn}"] = 0.0

    # 先按参数名存 network 结构
    basis_by_param = {}
    param_names = []

    for lmn_id in modes_use:
        # Bc
        name_cos = f"Bc{lmn_id}"
        basis_by_param[name_cos] = self._network_strain_single_mode_B(
            base_gw_params,
            lmn_id=lmn_id,
            Bc=1.0,
            Bs=0.0,
            network=network,
            channels=channels,
            TDIgen=TDIgen,
        )
        param_names.append(name_cos)

        # Bs
        name_sin = f"Bs{lmn_id}"
        basis_by_param[name_sin] = self._network_strain_single_mode_B(
            base_gw_params,
            lmn_id=lmn_id,
            Bc=0.0,
            Bs=1.0,
            network=network,
            channels=channels,
            TDIgen=TDIgen,
        )
        param_names.append(name_sin)

    # 重排成 basis_dict[det][ch] = [h_i]，与 param_names 对应
    basis_dict = {}
    for det in network:
        basis_dict[det] = {}
        for ch in channels:
            basis_dict[det][ch] = []

    for name in param_names:
        h_net = basis_by_param[name]
        for det in network:
            if det not in h_net:
                raise RuntimeError(f"基底 {name} 缺少探测器 {det}")
            for ch in channels:
                if ch not in h_net[det]:
                    raise RuntimeError(f"基底 {name} 缺少通道 {det}:{ch}")
                basis_dict[det][ch].append(h_net[det][ch])

    return basis_dict, param_names


def build_firefly_M_s(self, gw_params, modes=None, data=None, network=None, channels=None, TDIgen=1):
    """
    从 gw_params 直接构造 Firefly 所需的基底、M 和 s。

    参数
    ----
    gw_params : dict
        QNM 源参数。
    modes : list[str] or None
        使用的具体模式；None 时由 gw_params['lmns'] 展开。
    data : dict or None
        None -> s_i = <h_i|h_i>；
        dict -> s_i = <h_i|data>。
        data 的结构须为 data[det][ch] = array。
    network / channels / TDIgen
        与 build_firefly_basis 一致。

    返回
    ----
    basis_dict : dict[det][ch] = [h_i]
    param_names: list[str]
    M          : (Npar, Npar) 对称矩阵
    s          : (Npar,)
    """
    # 和 build_firefly_basis 一样处理 network / channels
    if network is None:
        network = self.det_list
    else:
        network = list(network)

    if channels is None:
        channels = self.channels
    else:
        channels = list(channels)

    # 先构造基底
    basis_dict, param_names = self.build_firefly_basis(
        gw_params,
        modes=modes,
        network=network,
        channels=channels,
        TDIgen=TDIgen,
    )

    npar = len(param_names)
    M = np.zeros((npar, npar))
    s_vec = np.zeros(npar)

    # 小 helper：从 basis_dict 取出第 i 个基底的 network 结构
    def get_hi(i):
        hi = {}
        for det in network:
            det_dict = {}
            for ch in channels:
                det_dict[ch] = basis_dict[det][ch][i]
            hi[det] = det_dict
        return hi

    # 构造 M 和 s（只算 i<=j）
    for i in range(npar):
        hi = get_hi(i)

        if data is None:
            s_vec[i] = self.network_inner_product(
                hi, hi, network=network, channels=channels
            )
        else:
            s_vec[i] = self.network_inner_product(
                hi, data, network=network, channels=channels
            )

        for j in range(i, npar):
            hj = get_hi(j)
            Mij = self.network_inner_product(
                hi, hj, network=network, channels=channels
            )
            M[i, j] = Mij
            if j != i:
                M[j, i] = Mij

    return M, s_vec
