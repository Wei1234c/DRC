# https://github.com/jaakkopasanen/AutoEq

from drc.measurements.frequency.responses import FrequencyResponse
from drc.tuners.response.equalizer import ResponseEqualizer


# from autoeq.frequency_response import FrequencyResponse
# from autoeq.constants import *
# from drc.tuners.response.peq import PEQs
# FrequencyResponse.FREQUENCY_TO_CENTER = 1000

measurement_file_path = "measurements/JamBox.csv"

# compensation_path = 'compensation/harman_over-ear_2018.csv'
compensation_path = 'compensation/zero.csv'

output_file_path = 'results/JamBox.csv'

measurement = FrequencyResponse.read_from_csv(measurement_file_path)
measurement.smoothen_fractional_octave()
measurement = FrequencyResponse('t', measurement.frequency, measurement.smoothed)
compensation = FrequencyResponse.read_from_csv(compensation_path)

measurement, peqs, n_peq_filters, peq_max_gains = \
    ResponseEqualizer.get_peq_filters(measurement,
                                      compensation,
                                      max_filters = [5,5],
                                      max_gain_dB = 6,
                                      # bass_boost_gain = 4
                                      )

measurement.plot_graph(show = True,
                       raw = True,
                       error = True,
                       smoothed = True,
                       error_smoothed = True,
                       equalization = True,
                       parametric_eq = True,
                       equalized = True,
                       target = True)

print(peqs.param_sets)
print(len(peqs.param_sets))

# def equalize(measurement_file_path, compensation_path, output_file_path):
#     measurement = FrequencyResponse.read_from_csv(measurement_file_path)
#     # measurement.interpolate()
#     # measurement.center()
#     # measurement.write_to_csv(measurement_file_path)
#
#     compensation = FrequencyResponse.read_from_csv(compensation_path)
#     compensation.interpolate()
#     compensation.center()
#
#     peq_filters, n_peq_filters, peq_max_gains, fbeq_filters, n_fbeq_filters, fbeq_max_gain = \
#         measurement.process(compensation = compensation,
#                             max_filters = 10, max_gain = 6,
#                             bass_boost_gain = 4)
#
#     measurement.write_eqapo_parametric_eq(output_file_path.replace('.csv', ' ParametricEQ.txt'),
#                                           peq_filters,
#                                           preamp = -(peq_max_gains[-1]))
#
#     peqs = PEQs()
#     for (freq, Q, gain_dB) in peq_filters:
#         peqs.add_peq(freq_Hz = freq, Q = Q, gain_dB = gain_dB, filter_type = 'peaking')
#
#     peqs.dump('peqs settings.txt')
#
#     # Write results to CSV file
#     measurement.write_to_csv(output_file_path)
#
#     # Write plots to file and optionally display them
#     measurement.plot_graph(show = True, close = not True, file_path = output_file_path.replace('.csv', '.png'), )
#
#     return peqs
#
#
#
# measurement_file_path = "measurements/JamBox.csv"
#
# # compensation_path = 'compensation/harman_over-ear_2018.csv'
# compensation_path = 'compensation/zero.csv'
#
# output_file_path = 'results/JamBox.csv'
#
# peqs = equalize(measurement_file_path, compensation_path, output_file_path)
#
# print(peqs.param_sets)
