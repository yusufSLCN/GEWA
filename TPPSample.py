import dataclasses
import numpy as np
import h5py

@dataclasses.dataclass
class TPPSample:
    simplified_mesh_path: str
    point_cloud_path: str
    pair_scores_path: str
    pair_score_matrix_path: str
    pair_grasps_save_path: str
    # normals_path:str
    grasps_path: str
    info: dict = None

    @property
    def point_cloud(self):
        return np.load(self.point_cloud_path)
    
    @property
    def pair_scores(self):
        return np.load(self.pair_scores_path)
    
    @property
    def pair_score_matrix(self):
        return np.load(self.pair_score_matrix_path)

    @property
    def pair_grasps(self):
        return np.load(self.pair_grasps_save_path, allow_pickle=True).item()
    
    @property
    def normals(self):
        return np.load(self.normals_path)
    
    @property
    def grasps(self):
        data = h5py.File(self.grasps_path, "r")
        grasp_poses = np.array(data["grasps/transforms"])
        grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        return grasp_poses, grasp_success
    

if __name__ == "__main__":
    sample = TPPSample(
        simplified_mesh_path="path/to/mesh.obj",
        point_cloud=np.random.rand(100, 3),
        point_grasp_idx=np.random.randint(0, 100, 10),
        approach_scores=np.random.rand(10),
        grasps_path="path/to/grasps.npy"
    )

    