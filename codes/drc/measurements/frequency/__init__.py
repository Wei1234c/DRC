import numpy as np
from matplotlib import pyplot as plt



class Octave:
    FREQ_BASE = 1e3
    FREQ_LIMS = (20, 20000)
    FC_5_BANDs = (100, 320, 1000, 3200, 10000)
    FC_10_BANDs = (31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
    FC_15_BANDs = (25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000)
    FC_31_BANDs = (20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                   1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000)


    @classmethod
    def get_octave_by_frac(cls, octave_frac = 3,
                           freq_min = min(FREQ_LIMS), freq_max = max(FREQ_LIMS),
                           freq_base = FREQ_BASE):
        octave_ratio = 2 ** (1 / octave_frac)
        boundry_divider = octave_ratio ** (1 / 2)
        octave_power_min = int(octave_frac * np.log2(freq_min / freq_base)) - 1
        octave_power_max = int(octave_frac * np.log2(freq_max / freq_base)) + 1

        freq_centre = freq_base * (2 ** (np.arange(octave_power_min, octave_power_max + 1) / octave_frac))

        freq_upper = freq_centre * boundry_divider
        freq_lower = freq_centre / boundry_divider
        Q = freq_centre / (freq_upper - freq_lower)

        bands = np.stack((freq_lower, freq_centre, freq_upper, Q), axis = 1)

        return bands


    @classmethod
    def get_octave_by_bands(cls, n_bands = 10, freq_min = min(FREQ_LIMS), freq_max = max(FREQ_LIMS)):
        octave_min = np.log2(freq_min)
        octave_max = np.log2(freq_max)
        octave_unit = (octave_max - octave_min) / (n_bands - 1)
        octaves = octave_min + np.arange(0, n_bands) * octave_unit

        freq_centre = 2 ** octaves
        freq_lower = 2 ** (octaves - octave_unit / 2)
        freq_upper = 2 ** (octaves + octave_unit / 2)
        Q = freq_centre / (freq_upper - freq_lower)

        bands = np.stack((freq_lower, freq_centre, freq_upper, Q), axis = 1)

        return bands



def plot_spectrum(fs, amps, figsize = (10, 5), freq_lims = (20, 22000)):
    plt.figure(figsize = figsize)
    plt.plot(fs, amps)
    plt.xscale('log')
    plt.xlim(freq_lims)
