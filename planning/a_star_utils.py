import math
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass, label, binary_erosion


FREE_SPACE = 0
TARGET = -1
MIN_SAFE_OBJ = 1
MAX_SAFE_OBJ = 5
MIN_UNSAFE_OBJ = 6
MAX_UNSAFE_OBJ = 10


DEFAULT_PARAMS = {
    "D_MAX": 25.0,
    "GAMMA": 1.3,
    "SAFETY_BUFFER_DISTANCE": 5,
    "HIGH_COST_THRESHOLD": 7.0,
    "SAFETY_BUFFER_PENALTY": 50.0,
}


def find_goal_location(base_map):
    coords = np.argwhere(base_map == TARGET)
    if coords.size == 0:
        raise ValueError("No target cell found in base map")
    centroid = coords.mean(axis=0)
    return tuple(centroid)


def normalized_safety_from_cost(cost_value):
    if cost_value <= 1:
        return 1.0
    if cost_value >= 10:
        return 0.0
    return max(0.0, min(1.0, (10.0 - cost_value) / 9.0))


def direction_to_angle_deg(vec):
    dx = vec[1]
    dy = -vec[0]
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    return ang


def angle_to_unit_vec(angle_deg):
    a = math.radians(angle_deg)
    ux = math.cos(a)
    uy = math.sin(a)
    return np.array([-uy, ux])


def estimate_normal_from_centroid(centroid, contact_point):
    if centroid is None or contact_point is None:
        return None
    try:
        vec = np.array(centroid) - np.array(contact_point)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return vec / norm
    except Exception:
        return None


def distance_to_target(pos, target_coords):
    return math.hypot(pos[0] - target_coords[0], pos[1] - target_coords[1])


def classify_cell(base_map, region_map, r, c, params=DEFAULT_PARAMS):
    if not (0 <= r < base_map.shape[0] and 0 <= c < base_map.shape[1]):
        return {'type': 'out', 'object_id': None, 'region_cost': float('inf'), 'is_high_cost_pixel': True}

    base_val = int(base_map[r, c])
    region_cost = float(region_map[r, c])

    is_high = region_cost > params["HIGH_COST_THRESHOLD"]

    if base_val == TARGET:
        return {'type': 'target', 'object_id': None, 'region_cost': region_cost, 'is_high_cost_pixel': is_high}
    if MIN_UNSAFE_OBJ <= base_val <= MAX_UNSAFE_OBJ:
        return {'type': 'unsafe_object', 'object_id': base_val, 'region_cost': region_cost, 'is_high_cost_pixel': is_high}
    if MIN_SAFE_OBJ <= base_val <= MAX_SAFE_OBJ:
        return {'type': 'safe_object', 'object_id': base_val, 'region_cost': region_cost, 'is_high_cost_pixel': is_high}
    return {'type': 'free', 'object_id': None, 'region_cost': region_cost, 'is_high_cost_pixel': is_high}


def create_safety_buffer_map(base_map, region_map, params=DEFAULT_PARAMS, include_safe_objects_as_hazards=False):
    buffer_dist = params["SAFETY_BUFFER_DISTANCE"]
    high_cost_thresh = params["HIGH_COST_THRESHOLD"]

    unsafe_mask = ((base_map >= MIN_UNSAFE_OBJ) & (base_map <= MAX_UNSAFE_OBJ))
    safe_obj_mask = ((base_map >= MIN_SAFE_OBJ) & (base_map <= MAX_SAFE_OBJ))
    high_cost_mask = (region_map > high_cost_thresh)
    if include_safe_objects_as_hazards:
        hazard_mask = unsafe_mask | safe_obj_mask | high_cost_mask
    else:
        hazard_mask = unsafe_mask | high_cost_mask

    safe_mask = ~hazard_mask
    distance_map = distance_transform_edt(safe_mask)
    safety_violation_map = distance_map < buffer_dist

    return safety_violation_map, distance_map


def check_safety_buffer_violation(footprint, safety_violation_map, distance_map, params=DEFAULT_PARAMS):
    violations = 0
    total_penalty = 0.0
    buffer_dist = params["SAFETY_BUFFER_DISTANCE"]

    for (r, c) in footprint:
        if not (0 <= r < safety_violation_map.shape[0] and 0 <= c < safety_violation_map.shape[1]):
            return True, float('inf')
        if safety_violation_map[r, c]:
            violations += 1
            dist_to_hazard = distance_map[r, c]
            if dist_to_hazard < 1e-6:
                return True, float('inf')
            penalty_factor = max(0, (buffer_dist - dist_to_hazard) / buffer_dist)
            total_penalty += params["SAFETY_BUFFER_PENALTY"] * (penalty_factor ** 2)

    violation_ratio = violations / len(footprint) if footprint else 0
    if violation_ratio > 0.3:
        return True, float('inf')

    return False, total_penalty


@lru_cache(maxsize=16384)
def get_gripper_footprint_cached(center_r, center_c, angle_deg, main_len=16, perp_len=4):
    return get_gripper_footprint_uncached(int(center_r), int(center_c), int(round(angle_deg)) % 360, main_len, perp_len)


def get_gripper_footprint_uncached(center_r, center_c, angle_deg, main_len=16, perp_len=4):
    footprint = set()
    half_main = main_len / 2.0
    half_perp = perp_len / 2.0
    pts = []
    x_min = -int(math.ceil(half_main))
    x_max = int(math.ceil(half_main))
    y_min = -int(math.ceil(half_perp))
    y_max = int(math.ceil(half_perp))
    for xi in range(x_min, x_max + 1):
        for yi in range(y_min, y_max + 1):
            if abs(xi) <= half_main and abs(yi) <= half_perp:
                pts.append((xi, yi))
    if angle_deg % 360 != 0:
        a = math.radians(angle_deg)
        ca = math.cos(a)
        sa = math.sin(a)
        rotated = []
        for x, y in pts:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            gx = int(round(rx))
            gy = int(round(ry))
            rotated.append((gx, gy))
        pts = rotated
    for dx, dy in pts:
        r = center_r + dy
        c = center_c + dx
        footprint.add((int(r), int(c)))
    return footprint


class ObjectState:
    def __init__(self, base_map, region_map):
        self.original_base_map = base_map.copy()
        self.current_base_map = base_map.copy()
        self.region_map = region_map
        self.safe_objects = {}
        labeled, _ = label(self.original_base_map > 0)
        max_label = int(np.max(labeled)) if labeled.size else 0
        for obj_id in range(1, max_label + 1):
            mask = (labeled == obj_id)
            if not np.any(mask):
                continue
            vals = np.unique(self.original_base_map[mask])
            val = int(vals[0]) if len(vals) > 0 else 1
            center = tuple(center_of_mass(mask))
            if not (np.isfinite(center[0]) and np.isfinite(center[1])):
                coords = np.argwhere(mask)
                if coords.size > 0:
                    center = tuple(coords[0])
                else:
                    continue
            self.safe_objects[obj_id] = {
                'mask': mask,
                'center': center,
                'value': val,
                'displaced_center': None
            }
        self.displaced_objects = {}
        self._displaced_mask_cache = {}

    def get_current_map(self):
        return self.current_base_map.copy()

    def _get_displaced_mask_for_obj(self, obj_id, new_center):
        key = (obj_id, int(round(new_center[0])), int(round(new_center[1])))
        if key in self._displaced_mask_cache:
            return self._displaced_mask_cache[key]
        info = self.safe_objects[obj_id]
        orig_mask = info['mask']
        orig_coords = np.argwhere(orig_mask)
        if orig_coords.size == 0:
            res = np.zeros_like(orig_mask, dtype=bool)
            self._displaced_mask_cache[key] = res
            return res
        ocenter = np.array(center_of_mass(orig_mask))
        if not (np.isfinite(ocenter[0]) and np.isfinite(ocenter[1])):
            ocenter = np.array(orig_coords[0])
        dr = int(round(new_center[0] - ocenter[0]))
        dc = int(round(new_center[1] - ocenter[1]))
        displaced = np.zeros_like(orig_mask, dtype=bool)
        for (r, c) in orig_coords:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < displaced.shape[0] and 0 <= nc < displaced.shape[1]:
                displaced[nr, nc] = True
        self._displaced_mask_cache[key] = displaced
        return displaced

    def apply_displacement(self, obj_id, new_center):
        self.displaced_objects[obj_id] = tuple(new_center)
        self.safe_objects[obj_id]['displaced_center'] = tuple(new_center)
        self.current_base_map = self.original_base_map.copy()
        for oid, nc in self.displaced_objects.items():
            disp_mask = self._get_displaced_mask_for_obj(oid, nc)
            self.current_base_map[self.safe_objects[oid]['mask']] = FREE_SPACE
            self.current_base_map[disp_mask] = int(self.safe_objects[oid]['value'])

    def copy(self):
        new = ObjectState(self.original_base_map, self.region_map)
        new.current_base_map = self.current_base_map.copy()
        new.displaced_objects = dict(self.displaced_objects)
        new._displaced_mask_cache = dict(self._displaced_mask_cache)
        for oid, nc in new.displaced_objects.items():
            new.safe_objects[oid]['displaced_center'] = nc
        return new


def mean_translation_for_angle(angle_rel_deg, params):
    th = math.radians(min(abs(angle_rel_deg), 90.0))
    d = params["D_MAX"] * (math.cos(th) ** params["GAMMA"])
    return max(0.0, d)


def sample_mean_displacement(contact_pt, centroid, push_angle_deg, params):
    normal_vec = estimate_normal_from_centroid(centroid, contact_pt)
    if normal_vec is None or np.linalg.norm(normal_vec) < 1e-8:
        angle_rel = 0.0
    else:
        normal_angle = direction_to_angle_deg(normal_vec)
        raw = abs((push_angle_deg - normal_angle + 180) % 360 - 180)
        angle_rel = min(raw, 180 - raw)
    d_mean = mean_translation_for_angle(angle_rel, params)
    uv = angle_to_unit_vec(push_angle_deg)
    disp = uv * d_mean
    return disp


def displaced_mask_for_center(mask, new_center):
    orig_coords = np.argwhere(mask)
    if orig_coords.size == 0:
        return np.zeros_like(mask, dtype=bool)
    ocenter = np.array(center_of_mass(mask))
    if not (np.isfinite(ocenter[0]) and np.isfinite(ocenter[1])):
        ocenter = np.array(orig_coords[0])
    dr = int(round(new_center[0] - ocenter[0]))
    dc = int(round(new_center[1] - ocenter[1]))
    res = np.zeros_like(mask, dtype=bool)
    for r, c in orig_coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]:
            res[nr, nc] = True
    return res


def quick_bbox_collision_check(state_map, mask_bool, end_center, params):
    rs, cs = np.where(mask_bool)
    if rs.size == 0:
        return False
    min_r, max_r = rs.min(), rs.max()
    min_c, max_c = cs.min(), cs.max()
    ocenter = np.array(center_of_mass(mask_bool))
    if not (np.isfinite(ocenter[0]) and np.isfinite(ocenter[1])):
        ocenter = np.array([rs[0], cs[0]])
    dr = int(round(end_center[0] - ocenter[0]))
    dc = int(round(end_center[1] - ocenter[1]))
    min_r2, max_r2 = min_r + dr, max_r + dr
    min_c2, max_c2 = min_c + dc, max_c + dc
    umin_r = max(0, min(min_r, min_r2) - params["BBOX_MARGIN"])
    umax_r = min(state_map.shape[0]-1, max(max_r, max_r2) + params["BBOX_MARGIN"])
    umin_c = max(0, min(min_c, min_c2) - params["BBOX_MARGIN"])
    umax_c = min(state_map.shape[1]-1, max(max_c, max_c2) + params["BBOX_MARGIN"])
    sub = state_map[umin_r:umax_r+1, umin_c:umax_c+1]
    if np.any(sub == TARGET):
        return False
    unsafe_mask = (sub >= MIN_UNSAFE_OBJ) & (sub <= MAX_UNSAFE_OBJ)
    if np.any(unsafe_mask):
        return False
    return True


def validate_push_target_safety(obj_mask, displacement, target_coords, params):
    orig_coords = np.argwhere(obj_mask)
    if orig_coords.size == 0:
        return True
    ocenter = np.array(center_of_mass(obj_mask))
    if not (np.isfinite(ocenter[0]) and np.isfinite(ocenter[1])):
        ocenter = np.array(orig_coords[0])
    dr = int(round(displacement[0]))
    dc = int(round(displacement[1]))
    for (r, c) in orig_coords:
        new_r, new_c = r + dr, c + dc
        dist_to_target = distance_to_target((new_r, new_c), target_coords)
        if dist_to_target <= params["TARGET_PROTECTION_RADIUS"]:
            return False
    return True


def reachable_distance(grid_map, start, goal):
    R, C = grid_map.shape
    s = (int(round(start[0])), int(round(start[1])))
    g = (int(round(goal[0])), int(round(goal[1])))
    if not (0 <= s[0] < R and 0 <= s[1] < C and 0 <= g[0] < R and 0 <= g[1] < C):
        return float('inf')
    visited = np.full(grid_map.shape, False)
    q = deque()
    q.append((s[0], s[1], 0))
    visited[s[0], s[1]] = True
    while q:
        r, c, d = q.popleft()
        if (r, c) == g:
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not visited[nr, nc]:
                cell = grid_map[nr, nc]
                if cell == FREE_SPACE or cell == TARGET:
                    visited[nr, nc] = True
                    q.append((nr, nc, d + 1))
    return float('inf')


def is_direct_path_blocked(current_pos, goal_pos, current_map, max_check_dist=20):
    start = np.array(current_pos)
    end = np.array(goal_pos)
    vec = end - start
    dist = np.linalg.norm(vec)
    if dist == 0:
        return False
    check_dist = min(dist, max_check_dist)
    steps = max(1, int(check_dist))
    unit_vec = vec / dist
    for i in range(1, steps + 1):
        check_pos = start + unit_vec * i
        r, c = int(round(check_pos[0])), int(round(check_pos[1]))
        if not (0 <= r < current_map.shape[0] and 0 <= c < current_map.shape[1]):
            continue
        cell_val = current_map[r, c]
        if cell_val != FREE_SPACE and cell_val != TARGET:
            return True
    return False


def object_boundary_coords(mask):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    boundary = mask & (~eroded)
    return np.argwhere(boundary)
