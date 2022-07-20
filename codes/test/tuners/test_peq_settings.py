from drc.filters.peq import PEQs


s = PEQs()

settings = '''
Filter Settings file

Room EQ V5.20.8
Dated: 2022/6/7 上午 11:26:51

Notes:

Equaliser: Generic
JamBox avg
Filter  1: ON  PK       Fc   321.0 Hz  Gain  -1.90 dB  Q  1.554
Filter  2: ON  PK       Fc   634.0 Hz  Gain  -3.60 dB  Q  1.819
Filter  3: ON  PK       Fc    1079 Hz  Gain   4.70 dB  Q  2.707
Filter  4: ON  PK       Fc    2384 Hz  Gain  -8.00 dB  Q  1.000
Filter  5: ON  None   
Filter  6: ON  None   
Filter  7: ON  None   
Filter  8: ON  None   
Filter  9: ON  None   
Filter 10: ON  None   
Filter 11: ON  None   
Filter 12: ON  None   
Filter 13: ON  None   
Filter 14: ON  None   
Filter 15: ON  None   
Filter 16: ON  None   
Filter 17: ON  None   
Filter 18: ON  None   
Filter 19: ON  None   
Filter 20: ON  None   
Filter 21: ON  None   
Filter 25: ON  None   
'''

s.loads(settings, field_sep = ' ', n_header_lines = 9, line_sep = '\n')

print(s.param_sets)

file_name = 'peq_settings.txt'

s.dump(file_name)
print(s.param_sets)

s.load(file_name)
print(s.param_sets)

print(s.get_coefficient_sets(n_filters = 7))


