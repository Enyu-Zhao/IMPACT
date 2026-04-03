import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math


SAFE_THRESHOLD = 5      
NUM_SAMPLES = 100   
SIGMA_D = 33.0       
SIGMA_THETA_DEG = 25.0  
USE_RANDOM_SAMPLING = True  


def flatten_voxel_grid(voxel_grid):
    voxel_trimmed = voxel_grid[:, :, 3:] 

    object_mask = (voxel_trimmed >= 1) & (voxel_trimmed <= 10)
    target_mask = voxel_trimmed == -1

    filtered_objects = np.where(object_mask, voxel_trimmed, 0)
    top_down_objects = filtered_objects.max(axis=2)

    target_presence = target_mask.any(axis=2)
    top_down_objects[target_presence] = -1  

    return top_down_objects


def visualize_2d_grid(grid, title="Top-down View"):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid[::-1], cmap='viridis', origin='lower')
    plt.colorbar(label="Object ID")
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.tight_layout()
    plt.show()


def process_voxel_scene(voxel_grid, visualize: bool = True):
    top_down_view = flatten_voxel_grid(voxel_grid)

    if visualize:
        visualize_2d_grid(top_down_view)

    return top_down_view


def find_objects(grid, safe_threshold):
    object_mask = (grid > 0)
    labeled_grid, num_objects = ndimage.label(object_mask)
    objects_info = {}
    
    if num_objects == 0:
        print("Warning: No objects found in the grid.")
        return labeled_grid, objects_info

    object_slices = ndimage.find_objects(labeled_grid)
    if object_slices is None:
        print("Warning: scipy.ndimage.find_objects returned None unexpectedly.")
        return labeled_grid, objects_info

    for obj_id in range(1, num_objects + 1):
        if obj_id - 1 >= len(object_slices) or object_slices[obj_id-1] is None:
            continue

        slc = object_slices[obj_id-1]
        obj_mask_full = (labeled_grid == obj_id)
        obj_grid_view = grid[slc]
        obj_label_view = labeled_grid[slc]
        obj_mask_view = (obj_label_view == obj_id)

        costs = np.unique(obj_grid_view[obj_mask_view])
        if len(costs) == 0:
            continue
        obj_cost = costs[0]

        coords = np.argwhere(obj_mask_full)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)

        structure = ndimage.generate_binary_structure(2, 1)
        eroded_mask = ndimage.binary_erosion(obj_mask_full, structure=structure, border_value=0)
        exterior_mask = obj_mask_full & ~eroded_mask
        exterior_points = np.argwhere(exterior_mask)

        if exterior_points.size == 0 and coords.size > 0:
            exterior_points = coords

        objects_info[obj_id] = {
            'cost': obj_cost, 
            'is_safe': obj_cost <= safe_threshold,
            'centroid': centroid, 
            'exterior_points': np.array(exterior_points),
            'mask': obj_mask_full, 
            'id': obj_id, 
            'labeled_value': obj_id
        }
    return labeled_grid, objects_info


def calculate_motion_vector(centroid, exterior_point):
    motion_vector = centroid - exterior_point
    magnitude = np.linalg.norm(motion_vector)
    if magnitude > 1e-6:
        motion_vector = motion_vector / magnitude
    else:
        motion_vector = np.array([0.0, 0.0])
    return motion_vector


def generate_sample_points(centroid, motion_vector, num_samples,
                         min_distance=20.0, max_distance=50.0,
                         random_sampling=False, sigma_d=25.0, sigma_theta_deg=20.0):
    samples = []
    angles_deg = []
    distances = []
    base_motion_angle_rad = math.atan2(motion_vector[0], motion_vector[1])

    for _ in range(num_samples):
        if random_sampling:
            # Uniform random sampling
            angle_offset_deg = np.random.uniform(low=-sigma_theta_deg, high=sigma_theta_deg)
            angle_offset_rad = angle_offset_deg * np.pi / 180.0
            angle_deg = angle_offset_deg
            distance = np.random.uniform(low=min_distance, high=max_distance)
            final_angle_rad = base_motion_angle_rad + angle_offset_rad
        else:
            # Gaussian sampling
            angle_offset_deg = np.random.normal(loc=0.0, scale=sigma_theta_deg)
            angle_offset_rad = angle_offset_deg * np.pi / 180.0
            angle_deg = angle_offset_deg
            distance = np.clip(np.random.exponential(scale=sigma_d), min_distance, max_distance)
            final_angle_rad = base_motion_angle_rad + angle_offset_rad

        dy = distance * math.sin(final_angle_rad)
        dx = distance * math.cos(final_angle_rad)
        displacement_vector = np.array([dy, dx])
        sample_point = centroid + displacement_vector
        samples.append(sample_point)
        angles_deg.append(abs(angle_deg))
        actual_distance = np.linalg.norm(sample_point - centroid)
        distances.append(actual_distance)

    return np.array(samples), angles_deg, distances


def check_collision(grid, labeled_grid, sample_point, max_radius, selected_obj_id, safe_threshold):
    y, x = int(round(sample_point[0])), int(round(sample_point[1]))
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return False, False, False

    max_radius_int = int(np.ceil(max_radius))
    y_min, y_max = max(0, y - max_radius_int), min(grid.shape[0], y + max_radius_int + 1)
    x_min, x_max = max(0, x - max_radius_int), min(grid.shape[1], x + max_radius_int + 1)
    region_y, region_x = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (region_y - sample_point[0])**2 + (region_x - sample_point[1])**2
    circular_mask = dist_sq <= max_radius**2

    cost_region = grid[y_min:y_max, x_min:x_max]
    label_region = labeled_grid[y_min:y_max, x_min:x_max]
    valid_collision_mask = circular_mask & (label_region != selected_obj_id)
    collided_costs = cost_region[valid_collision_mask]

    if collided_costs.size == 0:
        return False, False, False

    unsafe_collision = np.any(collided_costs > safe_threshold)
    safe_collision = np.any((collided_costs > 0) & (collided_costs <= safe_threshold))
    target_collision = np.any(collided_costs == -1)

    return unsafe_collision, safe_collision, target_collision


def calculate_movement_likelihood_gaussian(distance, angle_deg, sigma_d, sigma_theta_deg):
    if sigma_d <= 1e-6 or sigma_theta_deg <= 1e-6:
        return 0.0 if distance > 1e-6 or angle_deg > 1e-6 else 1.0
    if distance < 0: 
        distance = 0

    angle_rad = abs(angle_deg) * np.pi / 180.0
    sigma_theta_rad = sigma_theta_deg * np.pi / 180.0
    exponent_distance = (distance / sigma_d) ** 2
    exponent_angle = (angle_rad / sigma_theta_rad) ** 2
    likelihood = np.exp(-(exponent_distance + exponent_angle))
    return max(0.0, min(1.0, likelihood))


def evaluate_push_safety(grid, labeled_grid, obj_info, exterior_point, num_samples, 
                         sigma_d, sigma_theta_deg, use_random_sampling, safe_threshold):
    centroid = obj_info['centroid']
    motion_vector = calculate_motion_vector(centroid, exterior_point)
    
    if np.linalg.norm(motion_vector) < 1e-6:
        return 0.0
    
    # Calculate object radius
    distances_to_centroid = np.linalg.norm(obj_info['exterior_points'] - centroid, axis=1)
    max_radius = np.max(distances_to_centroid) if distances_to_centroid.size > 0 else 0.5
    max_radius = max(max_radius, 0.5)
    
    # Generate sample points
    samples, angles_deg, distances = generate_sample_points(
        centroid, motion_vector, num_samples,
        random_sampling=use_random_sampling,
        sigma_d=sigma_d, sigma_theta_deg=sigma_theta_deg
    )
    
    likelihoods = []
    position_safeties = []
    
    for i, sample in enumerate(samples):
        unsafe_coll, safe_coll, target_coll = check_collision(
            grid, labeled_grid, sample, max_radius, obj_info['labeled_value'], safe_threshold)
        
        likelihood = calculate_movement_likelihood_gaussian(
            distances[i], angles_deg[i], sigma_d, sigma_theta_deg
        )
        likelihoods.append(likelihood)
        
        if unsafe_coll: 
            position_safety = 0.1
        elif safe_coll: 
            position_safety = 0.6
        elif target_coll: 
            position_safety = 0.025
        else: 
            position_safety = 1.0
        position_safeties.append(position_safety)
    
    likelihoods_np = np.array(likelihoods)
    position_safeties_np = np.array(position_safeties)
    total_likelihood = np.sum(likelihoods_np)
    
    if total_likelihood > 1e-6:
        overall_safety = np.sum(likelihoods_np * position_safeties_np) / total_likelihood
    else:
        overall_safety = 0.0
    
    return overall_safety


def analyze_push_safety(voxel_grid=None, data_dir="", filename="", 
                       safe_threshold=SAFE_THRESHOLD, num_samples=NUM_SAMPLES,
                       sigma_d=SIGMA_D, sigma_theta_deg=SIGMA_THETA_DEG,
                       use_random_sampling=USE_RANDOM_SAMPLING, visualize=True):
    base_grid = process_voxel_scene(voxel_grid, visualize=False)

    # Find and analyze objects
    labeled_grid, objects_info = find_objects(base_grid, safe_threshold)
    if not objects_info: 
        raise ValueError("No objects found to analyze.")
    
    safe_objects_ids = [obj_id for obj_id, info in objects_info.items() if info['is_safe']]

    # Create cost map
    updated_grid = base_grid.copy().astype(float)
    
    print(f"--- Analyzing Push Safety for All Safe Objects ---")
    print(f"Found {len(safe_objects_ids)} safe objects (cost <= {safe_threshold})")
    
    total_exterior_points = 0
    updated_points = 0
    
    # Process each safe object
    for obj_id in safe_objects_ids:
        obj_info = objects_info[obj_id]
        exterior_points = obj_info['exterior_points']
        
        if exterior_points.size == 0:
            print(f"Safe object {obj_id} has no exterior points, skipping.")
            continue
        
        print(f"Processing Object ID: {obj_id} (Cost: {obj_info['cost']}) with {len(exterior_points)} exterior points")
        total_exterior_points += len(exterior_points)
        
        # Process each exterior point
        for ext_point in exterior_points:
            safety_score = evaluate_push_safety(
                base_grid, labeled_grid, obj_info, ext_point, num_samples,
                sigma_d, sigma_theta_deg, use_random_sampling, safe_threshold
            )
            
            # Update cost: blend original cost with safety-based cost
            new_cost = updated_grid[ext_point[0], ext_point[1]] * 0.25 + (10 - 10 * safety_score) * 0.75
            new_cost = max(1, min(10, new_cost))  # Clamp to [1, 10]
            
            updated_grid[ext_point[0], ext_point[1]] = new_cost
            updated_points += 1
    
    print(f"\n--- Results Summary ---")
    print(f"Processed {len(safe_objects_ids)} safe objects with {total_exterior_points} total exterior points")
    print(f"Updated {updated_points} points in the cost map")
   
    if visualize:
        # only one figure is created
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

        vmin = min(np.min(base_grid), np.min(updated_grid))
        vmax = max(np.max(base_grid), np.max(updated_grid))

        im1 = ax1.imshow(updated_grid, cmap='viridis', vmin=vmin, vmax=10)
        # ax1.set_title('Anisotropic Push Safety Adjusted Cost Map')
        fig.colorbar(im1, ax=ax1, label='Object Cost')
        # label size
        ax1.xaxis.label.set_size(16)
        ax1.yaxis.label.set_size(16)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        # legend font size
        cbar = im1.colorbar
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        output_filename = f"{data_dir}/{filename}_push_safety_updated_grid.png"
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Visualization saved to '{output_filename}'")
        plt.show()
    
    return base_grid, updated_grid
