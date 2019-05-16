import numpy as np

num_values = 10
min_value = 0
max_value = 1

def get_parameter_values(min_value, max_value):
    values = []
    for i in range(num_values):
        value = min_value + (max_value - min_value) * np.random.rand()
        values.append(value)
    return values
        
if __name__ == '__main__':
    values = get_parameter_values(min_value, max_value)
    with open("parameter_values.txt", 'w') as f:
        for value in values:
            f.write(str(value)+"\n")
