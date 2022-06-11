import pickle
import torch

from type_def import *


def get_init_params(local_dict: Dict[str, Any]):
    result_dict = {}
    for key, value in local_dict.items():
        if key[0] != '_' and key != 'self':
            result_dict[key] = value
    return result_dict


class UniversalUseModel:
    def __init__(self,
                 state_dict_path: str,
                 init_params_path: str,
                 model_class,
                 sample_to_train_input: Callable,
                 train_output_to_read_format: Callable):
        """
        默认是把模型放在cpu上了
        :param state_dict_path:
        :param init_params_path:
        :param model_class:
        :param sample_to_train_input:
        :param train_output_to_read_format:
        """
        self.state_dict_path = state_dict_path
        self.init_params_path = init_params_path
        self.sample_to_train_input = sample_to_train_input
        self.train_output_to_read_format = train_output_to_read_format

        init_params = pickle.load(open(self.init_params_path, 'rb'))
        self.model = model_class(**init_params)
        self.model.load_state_dict(torch.load(open(self.state_dict_path, 'rb'), map_location=torch.device('cpu')))

    def __call__(self, input_sample):
        train_input = self.sample_to_train_input(input_sample)
        output = self.model(**train_input)
        result = self.train_output_to_read_format(output)
        return result
