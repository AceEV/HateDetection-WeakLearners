import logging.config
from config.paths import logs_path
from datetime import datetime
from src.utility.common_functions import does_dir_exist, create_dir


def _get_logger():
	if not does_dir_exist(logs_path):
		create_dir(logs_path)
	logFilePath = logs_path + '/log_' + datetime.now().strftime('%d_%m_%Y') + '.log'
	logging.config.fileConfig(logs_path, defaults={'logfilename': logFilePath})
	logger = logging.getLogger("log1")
	return logger


logger = _get_logger()
