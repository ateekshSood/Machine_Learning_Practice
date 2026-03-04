dict = {'one' : 2,
        'two' : 1}

yeah = sorted(dict.items() ,  key=lambda x : x[1] , reverse=True)
keys_sort = [key[0] for key in yeah]
print(keys_sort)