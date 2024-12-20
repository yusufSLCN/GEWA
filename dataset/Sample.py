import dataclasses
import numpy as np
import h5py

@dataclasses.dataclass
class Sample:
    simplified_mesh_path: str
    point_cloud_path: str
    approach_score_path: str
    point_grasp_path: str
    grasps_path: str
    info: dict = None

    @property
    def point_cloud(self):
        return np.load(self.point_cloud_path)
    
    @property
    def approach_scores(self):
        return np.load(self.approach_score_path)
    
    @property
    def point_grasps(self):
        return np.load(self.point_grasp_path, allow_pickle=True)
    
    @property
    def grasps(self):
        data = h5py.File(self.grasps_path, "r")
        grasp_poses = np.array(data["grasps/transforms"])
        grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        return grasp_poses, grasp_success
    

if __name__ == "__main__":
    sample = Sample(
        simplified_mesh_path="path/to/mesh.obj",
        point_cloud=np.random.rand(100, 3),
        point_grasp_idx=np.random.randint(0, 100, 10),
        approach_scores=np.random.rand(10),
        grasps_path="path/to/grasps.npy"
    )

    