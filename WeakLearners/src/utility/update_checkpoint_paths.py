import os.path


def correct_paths(model_path, dir_paths, run_name):
	chpk_name, path_name, emp = model_path.split('"')
	model_name = path_name.split('/')[-1]
	new_path_name = chpk_name + '"' + dir_paths + '/' + run_name + '/' + model_name + '"'
	return new_path_name


def update_checkpoint_paths(model_path):
	dir_path = os.path.dirname(os.path.dirname(model_path))
	for run_name in os.listdir(dir_path):
		file_lines = []
		try:
			with open(dir_path + '/' + run_name + '/checkpoint', 'r') as f:
				for model_path in f.read().splitlines():
					file_lines.append(correct_paths(model_path, dir_path, run_name))
			with open(dir_path + '/' + run_name + '/checkpoint', 'w') as f:
				# print('Updating .... ' + str(file_lines))
				for line in file_lines:
					f.write(line)
					f.write('\n')
		except:
			continue
	print("Updated all the checkpoint_paths...")


if __name__ == "__main__":
	update_checkpoint_paths("/home/ayush/Desktop/pyCharm/predifix_data/output/models/multi_split_sent")

