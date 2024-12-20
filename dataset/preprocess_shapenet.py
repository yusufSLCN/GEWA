import subprocess
import os
from utils.acronym_utils import load_file_names, extract_sample_info


manifold_path = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/Manifold/build'

grasp_directory = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/grasps'
model_root = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/data/ShapeNetSem-backup/models-OBJ/models'

save_directory = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/data/simplified_obj'
grasp_file_names = load_file_names(grasp_directory)
sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)

# Loop through the commands and run them
for i, sample in  enumerate(sample_paths):
    if i < 8548:
        continue
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