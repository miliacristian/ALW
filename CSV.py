import numpy, csv


def read_csv(filecsv, skip_rows=0, delimiter=',', skip_column_left=0, skip_column_right=0, last_column_is_label=True,
             num_label=1):
    """
    Read file.csv filecsv with delimiter delimiter skipping the first skip_rows and skipping the first skip_column_left
     and the first skip_column_right from the left and the right respectively
    precondition:the labels are in the last columns
    :param filecsv: string,path to csv file
    :param skip_rows:int,row to skip
    :param last_column_is_label:boolean,if last_column_is_label==True the label must be read from last column(after
     remove the skip_column_right column from the right),otherwise is the first label(after remove the skip_column_left
      column from the left
    :return features set: X,labels set:Y
    """
    with open(filecsv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=delimiter)
        for i in range(skip_rows):
            next(readCSV)
        data = list(readCSV)
        for i in range(len(data)):
            data[i] = [x for x in data[i] if x]
        result = numpy.array(data)
        num_row = len(result)
        num_col = len(result[skip_rows])
    if last_column_is_label:
        X = result[:, skip_column_left:num_col - skip_column_right - num_label]  # data without columns of the labels
        Y = result[:, num_col - skip_column_right - num_label:]  # array of label
    else:  # first column is label
        X = result[:, 0 + skip_column_left + num_label:num_col - skip_column_right]
        Y = result[:,
            0 + skip_column_left:skip_column_left + num_label]  # array of label in the first num_label columns
    return X, Y


def convert_label_values(Y, list_old_label, list_new_label):
    """
    Convert label's value in label set Y from oldvalue(taken from list old_label)
     to new value(taken from list new_label)
    list_old_label[i] became list_new_label[i]
    :param Y:label set
    :param list_old_label: list of label to convert
    :param list_new_label: list of new label
    :return: Y convertito
    """
    for j in range(len(Y)):
        for i in range(len(list_old_label)):
            if Y[j] == list_old_label[i]:
                Y[j] = list_new_label[i]
    Y = convert_type_to_float(Y)
    return Y


def convert_type_to_float(data):
    """
    convert numeric values in float64,needed to scikit-learn library
    :param data:any,data to convert
    :return: data converted in float64
    """
    data = data.astype('float64')
    return data
