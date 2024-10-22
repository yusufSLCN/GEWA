from create_tpp_dataset import save_split_samples
import open3d as o3d

if __name__ == "__main__":
    train_samples, val_samples = save_split_samples('../data', 400, dataset_name="tpp_effdict_nomean_wnormals", contactnet_split=True)
    # print(train_paths[0])
    # print(val_paths[0])

    for sample in train_samples:
        simplified_obj_path = sample.simplified_mesh_path
        print(simplified_obj_path)
        origin_obj_path = sample.info['model_path']
        print(origin_obj_path)
        scale = sample.info['scale']
        print(scale)

        # Load the mesh
        origin_mesh = o3d.io.read_triangle_mesh(origin_obj_path)
        origin_mesh.scale(scale, center=origin_mesh.get_center())
        origin_mesh.compute_vertex_normals()
        origin_mesh.paint_uniform_color([0, 0, 1])
        center = origin_mesh.get_center()
        origin_mesh.translate(-center)
        origin_mesh.translate([0, -0.1, 0])

        simplified_mesh = o3d.io.read_triangle_mesh(simplified_obj_path)
        simplified_mesh.scale(scale, center=simplified_mesh.get_center())
        simplified_mesh.compute_vertex_normals()
        simplified_mesh.paint_uniform_color([1, 0, 0])
        center = simplified_mesh.get_center()
        simplified_mesh.translate(-center)
        simplified_mesh.translate([0, 0.1, 0])
        

        # Visualize the mesh
        o3d.visualization.draw_geometries([origin_mesh, simplified_mesh])


        #compute point cloud with poisson disk sampling
        pcd_orig = origin_mesh.sample_points_poisson_disk(1000)
        pcd_orig.paint_uniform_color([0, 0, 1])
        #move to origin
        # center = pcd_orig.get_center()
        # pcd_orig.translate(-center)
        # pcd_orig.translate([0, -0.1, 0])

        pcd_simplified = simplified_mesh.sample_points_poisson_disk(1000)
        pcd_simplified.paint_uniform_color([1, 0, 0])
        #move to origin
        # center = pcd_simplified.get_center()
        # pcd_simplified.translate(-center)
        # pcd_simplified.translate([0, 0.1, 0])


        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd_orig, pcd_simplified])       
