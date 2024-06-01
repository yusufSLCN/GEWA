from acronym_utils import get_simplified_samples

simplified_mesh_directory = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/data/simplified_obj'
simplified_samples = get_simplified_samples(simplified_mesh_directory)
print(len(simplified_samples))