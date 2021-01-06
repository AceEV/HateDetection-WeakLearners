from src.utility.common_functions import read_json, get_dir_path, dict2obj


def load_models_config(issue_id):
	config_dict = read_json(get_dir_path() + '/config/modelsConfig.dat')[str(issue_id)]
	config_dict['model_name'] = "run" + str(config_dict['run_no']) + \
								"_n" + str(config_dict['n_network']) + \
								"_b" + str(config_dict['batch_type'])
	params = dict2obj(**config_dict)
	return params, config_dict
