import numpy,csv
#file importato,per usare le funzioni usare nomefile.nomefunzione
#from Classifier import classifier from nomefile import nomefunzione

def read_csv(filecsv):
    """
    :param filecsv: string path to csv file
    :return features set: X,labels set:Y
    assunzione:le label solo nell'ultima colonna
    """
    with open(filecsv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data=list(readCSV)
        result=numpy.array(data)
        num_row=len(result)
        num_col=len(result[0])
    X = result[:,0:num_col-1]  # dati senza la colonna con le label
    Y= result[:,num_col-1] #array di label
    return X,Y

def convert_label_values(Y,list_old_label,list_new_label):
    """
    converte il valore delle label da un valore (old value preso dalla list_old_label) ad un nuovo valore(preso dalla lista list_new value)
    :param Y:label
    :param list_old_label: lista delle label da convertire
    :param list_new_label: lista delle nuove label
    :return: None
    list_old_label[i] diventa list_new_label[i]
    """
    for j in range(len(Y)):
        for i in range(len(list_old_label)):
            if(Y[j]==list_old_label[i]):
                Y[j]=list_new_label[i]
    Y=convert_type_to_float(Y)
    return Y

def convert_type_to_float(data):
    """
    converte i dati numerici in float64,necessari alla libreria scikitlearn
    :param data: dati da convertire
    :return: data convertiti in float64
    """
    data=data.astype('float64')
    return data