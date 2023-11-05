def getcolumns(filename):
    """
    Specifically designed to extract the two columns from Thomas Nikola's .dat files.
    Returns two columns, one with the wavelength and the other with the number of pixels.
    NOTE: Wavelength has units of mm.

    filename: Name of the file from which we are extracting the columns.
    """
    with open(filename,'r') as file:
        lines = file.readlines()


    data = [line.strip().split(',') for line in lines]
    
    col1 = [row[0] for row in data]
    col2 = [row[-1] for row in data]

    column1 = [float(x) for x in col1[1:]]
    column2 = [float(x) for x in col2[1:]]

    return column1,column2

def freq_get(column1):
    """
    Converts column1 to frequencies.
    """
    
    for i in range(len(column1)):
        column1[i] = (3e8)/(column1[i]*1e-3) #c = lambda*f, assuming lambda in mm.
        column1[i] = column1[i]/(1e9) #Convert frequency to Ghz.

    return column1
