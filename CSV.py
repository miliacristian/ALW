import numpy,csv
#file importato,per usare le funzioni usare nomefile.nomefunzione
#from Classifier import classifier from nomefile import nomefunzione

def read_csv(filecsv,skip_rows=0,delimiter=',',skip_column_left=0,skip_column_right=0):
    """
    Legge un file csv con delimitatore delimiter saltando le prime skip_rows righe e saltando le prime
    skip_column_left e skip_column_right colonne rispettivamente da sinistra e da destra
    da destra
    precondizioni:le label solo nell'ultima colonna dopo aver tolto le skip_column_right colonne a destra
    :param filecsv: string,path to csv file
    :param skip_rows:int,row to skip
    :param skip_column_left,skip_column_right:int column_left to skip,column_right_to skip
    :return features set: X,labels set:Y
    """
    with open(filecsv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=delimiter)
        for i in range(skip_rows):
            next(readCSV)
        data = list(readCSV)
        result=numpy.array(data)
        print(result)
        num_row=len(result)
        print("num_row",num_row)
        num_col=len(result[skip_rows])
        print("num_col",num_col)
    X = result[:,skip_column_left:num_col-skip_column_right-1]  # dati senza la colonna con le label
    Y= result[:,num_col-skip_column_right-1] #array di label
    return X,Y

def convert_label_values(Y,list_old_label,list_new_label):
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
            if(Y[j]==list_old_label[i]):
                Y[j]=list_new_label[i]
    Y=convert_type_to_float(Y)
    return Y

def convert_type_to_float(data):
    """
    converte i dati numerici in float64,necessari alla libreria scikit-learn
    :param data:any,dati da convertire
    :return: data convertiti in float64
    """
    data=data.astype('float64')
    return data