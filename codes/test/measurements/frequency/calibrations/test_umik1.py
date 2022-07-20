from drc.measurements.frequency.calibrations.miniDSP import UMIK1


umik = UMIK1()

# fn = 'UMIK-1 cal file 7103946.txt'
fn = 'UMIK-1 cal file 7103946_90deg.txt'

umik.load(file_name = fn, field_sep = None, n_header_lines = 2, line_sep = '\n')
# print(umik.responses)
print('serial_no', umik.serial_no)
print('gain_dB', umik.gain_dB)
print('sensitivity_factor_dBFS', umik.sensitivity_factor_dBFS )
print('sensitivity_dBFS', UMIK1.get_sensitivity_dBFS(sensitivity_factor_dBFS = umik.sensitivity_factor_dBFS,
                                                     gain_dB = 18))
print('sensitivity_dBFS', umik.sensitivity_dBFS)


file_name = 'umik.txt'
umik.dump(file_name)

umik.load(file_name)
print(umik.responses)
