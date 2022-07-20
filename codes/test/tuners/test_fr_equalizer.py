
from drc.filters.peq import PEQs

coeffs = PEQs.get_coefficients(freq_Hz = 321.0, Q = 1.554, gain_dB = -1.9, fs = 48000, filter_type = 'peaking')
print(list(coeffs))
coeffs = PEQs.get_coefficients(idx =2, freq_Hz = 321.0, Q = 1.554, gain_dB = -1.9, fs = 48000, filter_type = 'peaking')
print(list(coeffs))