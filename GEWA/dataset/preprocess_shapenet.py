import subprocess
import os
from utils.acronym_utils import load_file_names, extract_sample_info


manifold_path = '../Manifold/build'

grasp_directory = '../data/acronym/grasps'
model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
save_directory = '../data/simplified_obj'
grasp_file_names = load_file_names(grasp_directory)
sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Loop through the commands and run them
for i, sample in  enumerate(sample_paths):
    watertight_save_path = f'{save_directory}/temp_watertight_{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
    watertight_command = f'{manifold_path}/manifold {sample["model_path"]} {watertight_save_path} -s'

    simplify_save_path = f'{save_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
    simplify_command = f'{manifold_path}/simplify -i {watertight_save_path} -o {simplify_save_path} -m -r 0.02'
    if os.path.exists(simplify_save_path):
        print(f'{simplify_save_path} already exists')
        continue
    subprocess.run(watertight_command, shell=True)
    subprocess.run(simplify_command, shell=True)
    #delete file in watertight_save_path
    os.remove(watertight_save_path)
    print(f'Finished {i+1}/{len(sample_paths)}')