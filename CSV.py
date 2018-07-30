import numpy, csv


def read_csv(filecsv, skip_rows=0, delimiter=',', skip_column_left=0, skip_column_right=0, last_column_is_label=True,num_label=1):
    """
    Legge un file csv con delimitatore delimiter saltando le prime skip_rows righe e saltando le prime
    skip_column_left e skip_column_right colonne rispettivamente da sinistra e da destra
    da destra
    precondizioni:le label solo nell'ultima colonna
    :param filecsv: string,path to csv file
    :param skip_rows:int,row to skip
    :param last_column_is_label:boolean se last_column_is_label==True la lebel deve essere letta dall'ultima colonna
    (dopo aver tolto le skip_column_right colonne a destra),altrimenti è la prima label(dopo aver tolto le skip_column_left colonne a sinistra)
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
    print(result)
    exit(0)
    if last_column_is_label:
        X = result[:, skip_column_left:num_col - skip_column_right - 1]  # dati senza la colonna con le label
        Y = result[:, num_col - skip_column_right - 1]  # array di label
    else:
        X = result[:, 0 + skip_column_left + 1:num_col - skip_column_right]
        Y = result[:, 0 + skip_column_left]  # array di label nella prima colonna
    return X, Y


def convert_label_values(Y, list_old_label, list_new_label):
    """
    converte il valore delle label contentute in Y da un valore old value (preso dalla list_old_label) ad un nuovo valore new_value(preso dalla lista list_new value)
    list_old_label[i] diventa list_new_label[i]
    :param Y:label set
    :param list_old_label: lista delle label da convertire
    :param list_new_label: lista delle nuove label
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
    converte i dati numerici in float64,necessari alla libreria scikit-learn
    :param data:any,dati da convertire
    :return: data convertiti in float64
    """
    data = data.astype('float64')
    return data
