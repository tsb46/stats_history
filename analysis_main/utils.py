def flatten_nested_list(nested_list):
    # This function flattens a two-level nested list
    # Make nested list robust to 'Nones' in level two lists
    nested_list_nonull = [list_1 for list_1 in nested_list
                          if list_1 is not None]
    flattened_list = [entry for list in nested_list_nonull
                      for entry in list]
    return flattened_list


def remove_non_alphanumeric(string_input):
    # Remove non alphanumeric characters from a string
    return ''.join(c for c in string_input if c.isalnum())


