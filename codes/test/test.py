# # https://github.com/jaakkopasanen/AutoEq
#
# from drc.measurements.frequency.responses import FrequencyResponse
# from drc.tuners.response.equalizer import ResponseEqualizer
#
#
# # from autoeq.frequency_response import FrequencyResponse
# # from autoeq.constants import *
# # from drc.tuners.response.peq import PEQs
# # FrequencyResponse.FREQUENCY_TO_CENTER = 1000
#
# measurement_file_path = "./autoeq_test/results/JamBox.csv"
# measurement = FrequencyResponse.read_from_csv(measurement_file_path).get_smoothed()
#
# compensation_path = './autoeq_test/compensation/zero.csv'
# compensation = FrequencyResponse.read_from_csv(compensation_path)
#
# max_gain_dB = 12
# bass_boost_gain = 0
#
# measurement, peqs, n_fbeq_filters, fbeq_max_gain = \
#     ResponseEqualizer.get_fixed_band_filters(measurement,
#                                              compensation = compensation,
#                                              # fc = Octave.FC_31_BANDs,
#                                              max_gain_dB = max_gain_dB,
#                                              bass_boost_gain = bass_boost_gain)
#
# print(n_fbeq_filters)
#
# s = ResponseEqualizer.get_eqapo_graphic_eq(measurement,
#                                            compensation = compensation, )
# print(s)

from drc.measurements.frequency import Octave
print(Octave.get_octave_by_bands(5, 100, 10000))