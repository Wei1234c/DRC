from . import Response



class MLSSA(Response):

    # http://www.mlssa.com/
    # https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/systemschematicdesign/speakerresponsemlssa
    #  the 2nd line in the header of the matches the following exactly: <6 white spaces> “Hz” <2 white spaces> “Mag (dB)” <7 white spaces> “deg” <New line>

    def __init__(self, responses = None,
                 columns = f'      "Hz"  "Mag (dB)"       "deg"',
                 field_sep = ','):
        super().__init__(responses = responses, columns = columns, field_sep = field_sep)

