
def split_validation(path_list, validation_percentage):
    val_index = int(len(path_list) * (1 - validation_percentage))
    val = path_list[val_index:]
    training = path_list[0:(val_index - 1)]
    return training, val
