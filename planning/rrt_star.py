import numpy as np
from tqdm import tqdm
import random

from planning.rrt_utils import *
    

class RRTStarVoxelGrid:
    def __init__(self, cost_map, high_cost_threshold=5, step_size=10, goal_sample_rate=0.25, search_radius=100, goal_radius=0.1):
        self.voxel_grid = cost_map.voxel_grid
        self.voxel_size = cost_map.voxel_size
        self.origin = cost_map.origin
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.goal_radius = goal_radius
        self.best_path = None
        self.best_path_score = float('inf')
        self.x_dim, self.y_dim, self.z_dim = self.voxel_grid.shape
        self.distance_transform = compute_distance_transform(self.voxel_grid, high_cost_threshold, self.voxel_size)

        self.target_indices = np.argwhere(self.voxel_grid == -1)
        self.object_indices = np.argwhere(self.voxel_grid > 0)


    def find_path(self, start_pos, max_iterations=200, max_cost=10):
        nodes = [VoxelGridNode(world_to_grid(start_pos, self.voxel_size, self.origin))]
        paths_found = 0  

        for _ in tqdm(range(max_iterations)):
            sample = self.sample_pos()

            nearest_node = self.get_nearest_node(nodes, sample)
            new_position = steer(nearest_node.position, sample, self.step_size)
            
            if is_free_node(new_position, self.voxel_grid, max_cost):
                new_node = VoxelGridNode(new_position, nearest_node)
                near_nodes = self.get_near_nodes(nodes, new_position, self.search_radius)
                min_cost_node = nearest_node

                # Rewire tree to optimize costs based on nearby nodes
                for near_node in near_nodes:
                    node_distance = world_distance(new_position, near_node, self.voxel_size, self.origin, VoxelGridNode) / 100
                    if is_free_edge(new_node.position, near_node.position, self.voxel_grid, max_cost) and \
                            near_node.cost + node_distance < new_node.cost:
                        min_cost_node = near_node
                        new_node.cost = near_node.cost + node_distance
                
                new_node.parent = min_cost_node
                nodes.append(new_node)

                # Rewiring step for nearby nodes
                for near_node in near_nodes:
                    node_distance = world_distance(new_node, near_node, self.voxel_size, self.origin, VoxelGridNode) / 100
                    if near_node != new_node.parent and \
                            is_free_edge(new_node.position, near_node.position, self.voxel_grid, max_cost) and \
                            new_node.cost + node_distance < near_node.cost:
                        near_node.parent = new_node
                        near_node.cost = new_node.cost + node_distance
                
                # Check if we've reached the target position
                if self.intersects_target(new_node.position): 
                    path = reconstruct_path(new_node)
                    self.update_best_path(path)
                    paths_found += 1
            
            if paths_found > 0:
                print(f"[INFO] Paths found: {paths_found}")
            
        world_best_path = [grid_to_world(pos, self.voxel_size, self.origin).tolist() for pos in self.best_path]
        return world_best_path, self.best_path_score
    

    def intersects_target(self, node_position):
        x, y, z = node_position
        return self.voxel_grid[x, y, z] == -1
    

    def update_best_path(self, new_path, max_path_cost=20, max_distance=1.7, path_weight=0.7, distance_weight=0.3):
        new_path_cost = self.evaluate_path_cost(new_path)
        distance_to_obstacles = evaluate_distance_to_obstacles(new_path, self.distance_transform)

        if distance_to_obstacles == 0:
            distance_to_obstacles = 1e-6  # Small value to avoid division by zero

        normalized_path_cost = min(new_path_cost / max_path_cost, 1.0)
        normalized_distance = min(distance_to_obstacles / max_distance, 1.0)

        score = path_weight * normalized_path_cost + distance_weight * (1 - normalized_distance) 

        if score < self.best_path_score:
            self.best_path_score = score
            self.best_path = new_path


    def evaluate_path_cost(self, path):
        total_cost = 0
        for point in path:
            x, y, z = point
            cost = self.voxel_grid[x, y, z]
            total_cost += cost
        return total_cost


    def evaluate_distance_to_obstacles(self, path):
        return evaluate_distance_to_obstacles(path, self.distance_transform)
        

    def random_sample(self):
        x = np.random.randint(0, self.x_dim)
        y = np.random.randint(0, self.y_dim)
        z_limit = world_to_grid([0, 0, 0.3], self.voxel_size, self.origin)[2]
        z = np.random.randint(0, z_limit)
        return np.array([x, y, z])
    

    def sample_pos(self):
        if random.random() < self.goal_sample_rate:
            idx = np.random.choice(len(self.target_indices))
            sample = self.target_indices[idx]
        elif len(self.object_indices) > 0 and random.random() < self.goal_sample_rate + 0.3:
            idx = np.random.choice(len(self.object_indices))
            sample = self.object_indices[idx]
        else:
            sample = self.random_sample()
        return sample
    

    def get_nearest_node(self, nodes, point):
        return min(nodes, key=lambda node: distance(node.position, point))


    def get_near_nodes(self, nodes, point, radius):
        radius = int(radius)
        return [node for node in nodes if distance(node.position, point) < radius]

