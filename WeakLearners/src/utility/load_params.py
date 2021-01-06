from src.utility.common_functions import read_json, get_dir_path, dict2obj


def load_params(train, path=''):
	if train:
		params_dict = read_json(get_dir_path() + '/src/LSTMClassification/params.json')
	else:
		params_dict = read_json(path)
	params_dict['model_name'] = "run" + str(params_dict['run_no'])
	params = dict2obj(**params_dict)
	return params, params_dict
