from type_def import *


def get_init_params(local_dict: Dict[str, Any]):
    result_dict = {}
    for key, value in local_dict.items():
        if key[0] != '_' and key != 'self':
            result_dict[key] = value
    return result_dict
