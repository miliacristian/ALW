from numpy import argmax
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
def l():
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']#lista di label tipo:<class 'list'>
    values = array(data)
    print(values) #tipo:<class 'numpy.ndarray'>
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    print(type(onehot_encoded))
    # invert first example
    inverted = label_encoder.inverse_transform(integer_encoded)
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[1])])#Use `array.size > 0` to check that an array is not empty.
    print(inverted)