__author__ = "Johan Sabel, Felix Büttner, Joel Ekelöf"


import numpy as np


num_values = 10
min_log_exp = -5
max_log_exp = -1

def get_parameter_values(num_values, min_log_exp, max_log_exp):
    values = []
    for i in range(num_values):
        exp = min_log_exp + (max_log_exp - min_log_exp) * np.random.rand()
        value = 10**exp
        values.append(value)
    return values
        
if __name__ == '__main__':
    values = get_parameter_values(num_values, min_log_exp, max_log_exp)
    with open("parameter_values.txt", 'w') as f:
        for value in values:
            f.write(str(value)+"\n")
