def multiple_argument(*args):
    print(args)
    for arg in args:
        print(arg)
    return None
def multi_key_word_arg(l,**kwargs):
    for key, value in kwargs.items():
        print(key,value)
    return None

def l():
    print("prova")
    p='miao'
    multi_key_word_arg(p,b='blue',c='cyano',p='turc',)


