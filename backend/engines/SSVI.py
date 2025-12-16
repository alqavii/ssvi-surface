import numpy as np


class SSVICalibration:
    @staticmethod
    def _ssvi_surface_formula(K, F, theta, phi, rho):
        k = np.log(K / F)
        w = (
            1
            / 2
            * theta
            * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
        )

        return w
