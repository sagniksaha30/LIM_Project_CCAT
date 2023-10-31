def getcolumns(filename):
    """
    Specifically designed to extract the two columns from Thomas Nikola's .dat files.
    Returns two columns, one with the wavelength and the other with the number of pixels.
    NOTE: Wavelength has units of mm.

    filename: Name of the file from which we are extracting the columns.
    """
    with open(filename) as file:
        data = file.readlines()

    column_data = [line.split() for line in data]

    column1 = [line[0] for line in column_data][1:]
    column2 = [line[1] for line in column_data][1:]

    #There's a pesky comma at the end of each element. Need to remove that.
    for i in range(len(column1)):
        column1[i] = column1[i][:-1]
        column2[i] = column2[i][:-1]

    column1 = [float(x) for x in column1]
    column2 = [float(x) for x in column2]

    return column1,column2


def freq_time_get(column1,column2,time_each = 4000/15):
    """
    Converts column1 to frequencies. Converts column 2 to time to observe entire channel. Returns both columns.
    """
    for i in range(len(column1)):
        column1[i] = (3e8)/(column1[i]*1e-3) #c = lambda*f, assuming lambda in mm.
        column1[i] = column1[i]/(1e9) #Convert frequency to Ghz.

        column2[i] = column2[i]*time_each #Number of pixels * time observed each pixel.

    return column1,column2
