import unittest
from unittest.mock import patch

import numpy as np

from gwspace.response import get_AET_td
from gwspace.response_ringdown import get_AET_basis_td


class BasisWaveform:
    vec_k = np.array([0.0, 0.0, -1.0])
    modes_list = ["220", "330"]
    modes_dic = {"220": {"Y_lm_p": 0.8, "Y_lm_m": -0.3, "omega": 0.013},
                 "330": {"Y_lm_p": -0.4, "Y_lm_m": 0.6, "omega": 0.021}}
    coefficients = {"220": (0.7, -0.2), "330": (-0.4, 0.6)}

    def polarization(self):
        p_plus = np.diag([1.0, -1.0, 0.0])
        p_cross = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])
        return p_plus, p_cross

    def get_base_func(self, times, mode):
        times = np.asarray(times)
        phase = self.modes_dic[mode]["omega"]*times
        envelope = np.exp(-1e-4*times)
        return envelope*np.cos(phase), envelope*np.sin(phase)

    def get_hphc(self, times):
        h_plus = np.zeros_like(times)
        h_cross = np.zeros_like(times)

        for mode in self.modes_list:
            B1, B2 = self.coefficients[mode]
            meta = self.modes_dic[mode]
            # Independent complex-waveform expression for the direct response.
            h = (B1+1j*B2)*np.exp((-1e-4+1j*meta["omega"])*times)
            h_plus += meta["Y_lm_p"]*h.real
            h_cross += meta["Y_lm_m"]*h.imag

        return h_plus, h_cross


class RingdownResponseTests(unittest.TestCase):
    def test_aet_basis_reconstructs_direct_response(self):
        tf = np.linspace(1000.0, 5000.0, 32)

        for generation in (1, 2):
            with self.subTest(generation=generation):
                wf = BasisWaveform()
                direct = get_AET_td(wf, tf, det="TQ", TDIgen=generation)
                with patch.object(wf, "get_base_func", wraps=wf.get_base_func) as evaluate:
                    basis = get_AET_basis_td(wf, tf, det="TQ", TDIgen=generation)
                    self.assertEqual(evaluate.call_count, len(wf.modes_list))

                for channel, direct_channel in zip(("A", "E", "T"), direct):
                    reconstructed = np.zeros_like(direct_channel)
                    for mode, mode_basis in zip(wf.modes_list, basis):
                        B1, B2 = wf.coefficients[mode]
                        G1, G2 = mode_basis[channel]
                        reconstructed += B1*G1 + B2*G2

                    np.testing.assert_allclose(reconstructed, direct_channel,
                                               rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
