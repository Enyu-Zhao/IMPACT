import math
import heapq
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.ndimage import binary_dilation

from planning.a_star_utils import *


FREE_SPACE = 0
TARGET = -1
MIN_SAFE_OBJ = 1
MAX_SAFE_OBJ = 5
MIN_UNSAFE_OBJ = 6
MAX_UNSAFE_OBJ = 10

GRIPPER_MAIN_AXIS_LENGTH = 16
GRIPPER_PERP_AXIS_LENGTH = 4

PLANNER_PARAMS = {
    # push physics model
    "D_MAX": 25.0,
    "GAMMA": 1.3,
    "SIGMA_D_BASE": 3.0,
    "ANGLE_BINS": 8,
    "STEP_SIZE": 2.0,
    # costs and safety
    "PUSH_BASE_COST": 8.0,
    "MOVE_COST": 1.0,
    "ROTATION_COST": 1.8,
    "SAFETY_SCORE_REJECT": 0.15,
    "SAFETY_SCORE_MIN_ACCEPT": 0.3,
    "SAFETY_WEIGHT": 45.0,
    "DISTURBANCE_WEIGHT": 3.0,
    "REACH_GAIN_BONUS": 15.0,
    "MAX_PUSH_OPTIONS_PER_POINT": 5,
    # SAFETY DISTANCE
    "SAFETY_BUFFER_DISTANCE": 5,
    "HIGH_COST_THRESHOLD": 7.0,     # region_map > this is considered high-cost pixel
    "SAFETY_BUFFER_PENALTY": 50.0,
    # footprint / region constraints
    "MAX_REGION_COST_FOR_GRIPPER": 6.0,
    "REGION_COST_MULTIPLIER": 8.0,
    "MIN_DIST_UNSAFE": 3,
    # pruning and pre-checks
    "PUSH_ANGLE_ALLOWANCE_DEG": 60,
    "BBOX_MARGIN": 1,
    # target protection
    "TARGET_PROTECTION_RADIUS": 3,
    "OBJECT_COLLISION_PENALTY": 80.0,
    # AGGRESSIVE PUSHING
    "BLOCKED_PATH_PUSH_BONUS": 20.0,
    "PUSH_EXPLORATION_RADIUS": 8,
    "MIN_PUSH_DISTANCE": 2.0,
    
    "STRICT_HIGH_COST_BUFFER": True,
    "HIGH_COST_STRICT_DISTANCE": 8,
}

@lru_cache(maxsize=16384)
def get_gripper_footprint_cached(center_r, center_c, angle_deg, main_len=GRIPPER_MAIN_AXIS_LENGTH, perp_len=GRIPPER_PERP_AXIS_LENGTH):
    return get_gripper_footprint_uncached(int(center_r), int(center_c), int(round(angle_deg)) % 360, main_len, perp_len)


def get_gripper_footprint_uncached(center_r, center_c, angle_deg, main_len=GRIPPER_MAIN_AXIS_LENGTH, perp_len=GRIPPER_PERP_AXIS_LENGTH):
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
        ca = math.cos(a); sa = math.sin(a)
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

def swept_check_object_and_gripper(state_map, obj_info, start_center, end_center, gripper_start, gripper_end, target_coords, safety_violation_map, distance_map, params=PLANNER_PARAMS):
    mask = obj_info['mask']
    if not quick_bbox_collision_check(state_map, mask, end_center, params):
        return False, 0.0, float('inf')
    displacement = np.array(end_center) - np.array(start_center)
    if not validate_push_target_safety(mask, displacement, target_coords, params):
        return False, 0.0, float('inf')
    vec = np.array(end_center) - np.array(start_center)
    dist = np.linalg.norm(vec)
    step = max(1, params["STEP_SIZE"])
    steps = int(math.ceil(dist / step)) if dist > 0 else 1
    disturbance = 0.0
    for i in range(1, steps + 1):
        t = i / float(steps)
        mid_center = np.array(start_center) + vec * t
        disp_mask = displaced_mask_for_center(mask, mid_center)
        mid_gripper = (
            gripper_start[0] + (gripper_end[0] - gripper_start[0]) * t,
            gripper_start[1] + (gripper_end[1] - gripper_start[1]) * t,
            gripper_start[2] + (gripper_end[2] - gripper_start[2]) * t,
        )
        gr_fp = get_gripper_footprint_cached(int(round(mid_gripper[0])), int(round(mid_gripper[1])), int(round(mid_gripper[2])))
        # Check safety buffer for gripper during sweep
        buffer_violation, _ = check_safety_buffer_violation(gr_fp, safety_violation_map, distance_map, params)
        if buffer_violation:
            return False, 0.0, float('inf')
        rs, cs = np.where(disp_mask)
        for r, c in zip(rs, cs):
            if not (0 <= r < state_map.shape[0] and 0 <= c < state_map.shape[1]):
                return False, 0.0, float('inf')
            v = state_map[r, c]
            if v == TARGET:
                return False, 0.0, float('inf')
            if MIN_UNSAFE_OBJ <= v <= MAX_UNSAFE_OBJ:
                return False, 0.0, float('inf')
        for (gr, gc) in gr_fp:
            if not (0 <= gr < state_map.shape[0] and 0 <= gc < state_map.shape[1]):
                return False, 0.0, float('inf')
            v = state_map[gr, gc]
            if MIN_UNSAFE_OBJ <= v <= MAX_UNSAFE_OBJ:
                return False, 0.0, float('inf')
        disturbance += dist / float(steps)
    return True, disturbance, 0.0


class Node:
    def __init__(self, r, c, angle, obj_state, g=1e9, h=0, parent=None, action=None):
        self.r = int(r); self.c = int(c)
        self.angle = float(angle)
        self.obj_state = obj_state
        self.g = g; self.h = h; self.f = g + h
        self.parent = parent
        self.action = action
    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f
    def key(self):
        ang_sym = int(round(self.angle)) % 180
        disp = tuple(sorted(self.obj_state.displaced_objects.items()))
        return (self.r, self.c, ang_sym, disp)


def heuristic(r, c, goal_r, goal_c):
    return math.hypot(r - goal_r, c - goal_c)


def get_actions(node, grid_shape, region_map, target_coords, safety_violation_map, distance_map):
    actions = []
    current_map = node.obj_state.get_current_map()
    object_boundary_cache = {}

    path_blocked = is_direct_path_blocked((node.r, node.c), target_coords, current_map)
    to_goal_vec = np.array(target_coords) - np.array([node.r, node.c])
    to_goal_perp_unit = None
    to_goal_norm = np.linalg.norm(to_goal_vec)
    if to_goal_norm > 0:
        to_goal_unit = to_goal_vec / to_goal_norm
        to_goal_perp_unit = np.array([-to_goal_unit[1], to_goal_unit[0]])

    # Movement actions: allow touching safe objects (even at high-cost pixels) but forbid unsafe objects
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = node.r + dr, node.c + dc
            if not (0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]):
                continue
            # For movement: validate gripper footprint, but allow safe object high-cost pixels (do not treat them as unsafe)
            # validate_gripper_placement returns False for absolute no-go (unsafe object or too-high region) — we keep that
            valid, penalty = validate_gripper_placement(nr, nc, node.angle, current_map, region_map, safety_violation_map, distance_map)
            if not valid:
                continue
            # However: treat safe-object-high-cost pixels as allowable for movement (they won't have caused valid==False)
            base_cost = math.hypot(dr, dc) * PLANNER_PARAMS["MOVE_COST"]
            total_cost = base_cost + penalty
            actions.append(('move', (nr, nc, node.angle), total_cost, None))

    # Rotations
    for da in [-45, 45]:
        new_ang = (((node.angle + da) + 180) % 360) - 180
        valid, penalty = validate_gripper_placement(node.r, node.c, new_ang, current_map, region_map, safety_violation_map, distance_map)
        if not valid:
            continue
        actions.append(('rotate', (node.r, node.c, new_ang), PLANNER_PARAMS["ROTATION_COST"] + penalty, None))

    # Push generation: more aggressive but with the strict rule:
    # -> If contact point is a high-cost pixel (region_map > HIGH_COST_THRESHOLD), reject push from that point.
    push_radius = PLANNER_PARAMS["PUSH_EXPLORATION_RADIUS"]
    if path_blocked:
        push_radius = int(push_radius * 1.5)
    grid_r, grid_c = grid_shape
    rmin = max(0, node.r - push_radius); rmax = min(grid_r - 1, node.r + push_radius)
    cmin = max(0, node.c - push_radius); cmax = min(grid_c - 1, node.c + push_radius)

    push_candidates = []
    for rr in range(rmin, rmax + 1):
        for cc in range(cmin, cmax + 1):
            if math.hypot(rr - node.r, cc - node.c) > push_radius:
                continue
            safety_cost = region_map[rr, cc]
            safety_score = normalized_safety_from_cost(safety_cost)
            if safety_score < PLANNER_PARAMS["SAFETY_SCORE_REJECT"]:
                continue
            val = current_map[rr, cc]
            if val == FREE_SPACE or val == TARGET:
                continue
            # only consider safe object pushes (we don't push unsafe objects)
            if not (MIN_SAFE_OBJ <= val <= MAX_SAFE_OBJ):
                continue
            obj_id = int(val)
            obj_info = node.obj_state.safe_objects.get(obj_id, None)
            if obj_info is None:
                continue
            centroid = obj_info.get('displaced_center', obj_info['center'])
            if centroid is None:
                continue
            # compute push candidate value (prefer objects aligned with goal when blocked)
            push_value = 0.0
            if path_blocked:
                to_goal = np.array(target_coords) - np.array([node.r, node.c])
                to_obj = np.array([rr, cc]) - np.array([node.r, node.c])
                if np.linalg.norm(to_goal) > 0 and np.linalg.norm(to_obj) > 0:
                    alignment = np.dot(to_goal, to_obj) / (np.linalg.norm(to_goal) * np.linalg.norm(to_obj))
                    if alignment > 0.3:
                        push_value += PLANNER_PARAMS["BLOCKED_PATH_PUSH_BONUS"]
            dist_to_gripper = math.hypot(rr - node.r, cc - node.c)
            push_value += max(0, 10 - dist_to_gripper)
            push_candidates.append((push_value, rr, cc, safety_score, obj_id, obj_info, centroid))

    push_candidates.sort(reverse=True, key=lambda x: x[0])
    max_pushes = PLANNER_PARAMS["MAX_PUSH_OPTIONS_PER_POINT"] * 2 if path_blocked else PLANNER_PARAMS["MAX_PUSH_OPTIONS_PER_POINT"]

    for push_value, rr, cc, safety_score, obj_id, obj_info, centroid in push_candidates[:max_pushes]:
        # For pushes, enforce the strict rule: contact cell must NOT be high-cost pixel
        cell_info = classify_cell(current_map, region_map, rr, cc)
        if cell_info['is_high_cost_pixel']:
            # Skip generating pushes from high-cost pixel even if it belongs to safe object
            continue

        # require contact point to be on object boundary (not interior)
        bset = object_boundary_cache.get(obj_id)
        if bset is None:
            boundary_coords = object_boundary_coords(obj_info['mask'])
            bset = set((int(r), int(c)) for r, c in boundary_coords)
            object_boundary_cache[obj_id] = bset
        if (rr, cc) not in bset:
            # not a boundary contact -> skip push generation (interior brushing allowed, but push not)
            continue

        # Generate directions
        normal_vec = estimate_normal_from_centroid(centroid, (rr, cc))
        dir_candidates = []
        if normal_vec is None:
            dir_candidates = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            normal_ang = direction_to_angle_deg(normal_vec)
            allow = PLANNER_PARAMS["PUSH_ANGLE_ALLOWANCE_DEG"]
            offsets = [0, -allow/3.0, allow/3.0, -allow/2.0, allow/2.0, -allow, allow]
            for off in offsets:
                dir_candidates.append((normal_ang + off) % 360)
            dir_candidates.extend([node.angle % 360, (node.angle + 90) % 360, (node.angle - 90) % 360])
        angle_scores = []
        for ang in dir_candidates:
            score = push_value
            uv = angle_to_unit_vec(ang)
            test_pos = np.array([rr, cc]) + uv * 5
            tr, tc = int(round(test_pos[0])), int(round(test_pos[1]))
            if 0 <= tr < grid_r and 0 <= tc < grid_c and current_map[tr, tc] == FREE_SPACE:
                score += 5.0
            if to_goal_perp_unit is not None:
                perp_alignment = abs(np.dot(uv, to_goal_perp_unit))
                score += perp_alignment * 3.0
            angle_scores.append((score, ang))
        angle_scores.sort(reverse=True, key=lambda x: x[0])
        selected_angles = [ang for _, ang in angle_scores[:3]]
        for ang in selected_angles:
            push_meta = {'contact': (rr, cc), 'angle': ang, 'safety_score': safety_score, 'push_value': push_value, 'obj_id': obj_id}
            base_cost = PLANNER_PARAMS["PUSH_BASE_COST"]
            safety_penalty = (1.0 - safety_score) * PLANNER_PARAMS["SAFETY_WEIGHT"]
            if push_value > 15:
                base_cost *= 0.7
            total_cost = base_cost + safety_penalty
            actions.append(('push', (node.r, node.c, node.angle), total_cost, push_meta))

    return actions


def validate_gripper_placement(gripper_r, gripper_c, gripper_angle,
                                base_map, region_map, safety_violation_map, distance_map,
                                params=PLANNER_PARAMS):
    fp = get_gripper_footprint_cached(gripper_r, gripper_c, gripper_angle)
    total_region_cost = 0.0
    total_object_penalty = 0.0

    # STRICT RULE FOR HIGH-COST OBJECTS
    if params.get("STRICT_HIGH_COST_BUFFER", True):
        high_cost_mask = region_map > params["HIGH_COST_THRESHOLD"]
        # Expand high-cost areas by fixed distance (3 pixels here)
        strict_buffer = binary_dilation(high_cost_mask,
                                        iterations=params.get("HIGH_COST_STRICT_DISTANCE", 3))
        for (fr, fc) in fp:
            if 0 <= fr < base_map.shape[0] and 0 <= fc < base_map.shape[1]:
                if strict_buffer[fr, fc]:
                    return False, float('inf') 

    # existing safety buffer check
    buffer_violation, buffer_penalty = check_safety_buffer_violation(fp, safety_violation_map, distance_map, params)
    if buffer_violation:
        return False, float('inf')

    for (fr, fc) in fp:
        if not (0 <= fr < base_map.shape[0] and 0 <= fc < base_map.shape[1]):
            return False, float('inf')
        base_val = int(base_map[fr, fc])
        if MIN_UNSAFE_OBJ <= base_val <= MAX_UNSAFE_OBJ:
            return False, float('inf')
        if MIN_SAFE_OBJ <= base_val <= MAX_SAFE_OBJ:
            total_object_penalty += params["OBJECT_COLLISION_PENALTY"]
        region_cost = region_map[fr, fc]
        total_region_cost += region_cost
        if region_cost > params["MAX_REGION_COST_FOR_GRIPPER"]:
            return False, float('inf')

    avg_region_cost = total_region_cost / len(fp) if fp else 0
    region_penalty = avg_region_cost * params["REGION_COST_MULTIPLIER"]
    total_penalty = region_penalty + total_object_penalty + buffer_penalty
    return True, total_penalty


def directional_contact_planner(start_coords, start_angle, goal_coords, base_map, region_map):
    print("Planner: strict-push-rules a_star2. Params:", PLANNER_PARAMS)
    safety_violation_map, distance_map = create_safety_buffer_map(base_map, region_map)
    grid_shape = base_map.shape
    goal_r, goal_c = goal_coords
    initial_state = ObjectState(base_map, region_map)
    valid_start, _ = validate_gripper_placement(start_coords[0], start_coords[1], start_angle,
                                                initial_state.get_current_map(), region_map, safety_violation_map, distance_map)
    if not valid_start:
        print("ERROR: Start violates safety constraints")
        return None, float('inf')
    start_node = Node(start_coords[0], start_coords[1], start_angle, initial_state, g=0,
                      h=heuristic(start_coords[0], start_coords[1], goal_r, goal_c))
    open_heap = []
    heapq.heappush(open_heap, start_node)
    closed = dict()
    nodes = 0
    while open_heap:
        nodes += 1
        if nodes % 200 == 0:
            print(f"[DEBUG] Expanded {nodes} nodes, open={len(open_heap)}, closed={len(closed)}")
        cur = heapq.heappop(open_heap)
        cmap = cur.obj_state.get_current_map()
        cr, cc = cur.r, cur.c
        if 0 <= cr < grid_shape[0] and 0 <= cc < grid_shape[1] and cmap[cr, cc] == TARGET:
            print(f"Goal reached after {nodes} nodes")
            return reconstruct_path(cur), float(cur.g)
        key = cur.key()
        if key in closed and closed[key] <= cur.g:
            continue
        closed[key] = cur.g
        actions = get_actions(cur, grid_shape, region_map, goal_coords, safety_violation_map, distance_map)
        for act_type, next_conf, base_cost, meta in actions:
            nr, nc, nang = next_conf
            new_state = cur.obj_state.copy()
            action_description = ""
            success = True
            added_cost = 0.0
            if act_type == 'move':
                action_description = f"move to ({nr}, {nc})"
            elif act_type == 'rotate':
                action_description = f"rotate to {nang:.1f}°"
            elif act_type == 'push':
                contact = meta['contact']; push_ang = meta['angle']
                safety_score = meta['safety_score']; push_value = meta['push_value']; obj_id = meta['obj_id']
                cm = new_state.get_current_map()
                cr_val = cm[contact[0], contact[1]]
                if cr_val == FREE_SPACE or cr_val == TARGET or int(cr_val) != obj_id:
                    success = False
                else:
                    obj_info = new_state.safe_objects.get(obj_id, None)
                    if obj_info is None:
                        success = False
                    else:
                        start_center = obj_info.get('displaced_center', obj_info['center'])
                        if start_center is None:
                            success = False
                        else:
                            mean_disp = sample_mean_displacement(contact, start_center, push_ang, PLANNER_PARAMS)
                            displacement_dist = np.linalg.norm(mean_disp)
                            if displacement_dist < PLANNER_PARAMS["MIN_PUSH_DISTANCE"]:
                                success = False
                            else:
                                end_center = (start_center[0] + mean_disp[0], start_center[1] + mean_disp[1])
                                gripper_start = (cur.r, cur.c, cur.angle)
                                gripper_end = (gripper_start[0] + mean_disp[0], gripper_start[1] + mean_disp[1], cur.angle)
                                ok, disturbance, _ = swept_check_object_and_gripper(new_state.get_current_map(), obj_info, start_center, end_center,
                                                                                   gripper_start, gripper_end, goal_coords, safety_violation_map, distance_map)
                                if not ok:
                                    success = False
                                else:
                                    tmp_state = new_state.copy()
                                    tmp_state.apply_displacement(obj_id, end_center)
                                    old_dist = reachable_distance(new_state.get_current_map(), (cur.r, cur.c), (goal_r, goal_c))
                                    new_dist = reachable_distance(tmp_state.get_current_map(), (cur.r, cur.c), (goal_r, goal_c))
                                    reach_gain = max(0.0, old_dist - new_dist)
                                    safety_penalty = (1.0 - safety_score) * PLANNER_PARAMS["SAFETY_WEIGHT"]
                                    disturbance_penalty = disturbance * PLANNER_PARAMS["DISTURBANCE_WEIGHT"]
                                    push_cost = PLANNER_PARAMS["PUSH_BASE_COST"] + safety_penalty + disturbance_penalty
                                    if reach_gain > 0.5:
                                        bonus = PLANNER_PARAMS["REACH_GAIN_BONUS"] * min(reach_gain, 8.0)
                                        push_cost -= bonus
                                    if push_value > 15:
                                        push_cost -= 5.0
                                    added_cost = max(1.0, push_cost)
                                    new_state.apply_displacement(obj_id, end_center)
                                    action_description = f"push obj{obj_id} @{contact} ang={push_ang:.0f}° disp={displacement_dist:.1f} val={push_value:.1f}"
            if not success:
                continue
            new_map = new_state.get_current_map()
            valid_fp, fp_penalty = validate_gripper_placement(nr, nc, nang, new_map, region_map, safety_violation_map, distance_map)
            if not valid_fp or fp_penalty == float('inf'):
                continue
            tentative_g = cur.g + base_cost + added_cost + fp_penalty
            h = heuristic(nr, nc, goal_r, goal_c)
            new_node = Node(nr, nc, nang, new_state, g=tentative_g, h=h, parent=cur, action=action_description)
            new_key = new_node.key()
            if new_key in closed and closed[new_key] <= tentative_g:
                continue
            heapq.heappush(open_heap, new_node)
    print("No path found")
    return None, float('inf')


def reconstruct_path(goal_node):
    path = []
    c = goal_node
    while c is not None:
        path.append({
            'position': (c.r, c.c),
            'angle': c.angle-180,
            'object_state': c.obj_state.displaced_objects.copy(),
            'action': c.action
        })
        c = c.parent
    path.reverse()
    return path


def visualise_path(base_map, region_map, path, start_coords, goal_coords, total_cost, data_dir="", filename=""):
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    safety_violation_map, distance_map = create_safety_buffer_map(base_map, region_map)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(region_map, cmap='viridis', origin='lower', alpha=0.8)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Region cost')
    safety_overlay = np.ma.masked_where(~safety_violation_map, np.ones_like(safety_violation_map))
    ax1.imshow(safety_overlay, cmap='Reds', alpha=0.4, origin='lower')
    ax1.set_title("Region map + Safety buffer zones (red)")
    ax2 = axes[0, 1]
    im2 = ax2.imshow(distance_map, cmap='plasma', origin='lower')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Distance to hazards')
    ax2.set_title("Distance to hazards map")
    ax3 = axes[1, 0]
    im3 = ax3.imshow(base_map, cmap='tab20', origin='lower')
    ax3.set_title("Object map")
    ax4 = axes[1, 1]
    im4 = ax4.imshow(region_map, cmap='viridis', origin='lower', alpha=0.6)
    ax4.set_title(f"Planned path (cost: {total_cost:.1f})")

    for ax in [ax1, ax2, ax3, ax4]:
        ur, uc = np.where((base_map >= MIN_UNSAFE_OBJ) & (base_map <= MAX_UNSAFE_OBJ))
        ax.scatter(uc, ur, s=25, marker='x', c='red', label='Unsafe')
        sr, sc = np.where((base_map >= MIN_SAFE_OBJ) & (base_map <= MAX_SAFE_OBJ))
        ax.scatter(sc, sr, s=15, marker='o', c='green', alpha=0.6, label='Safe')
        ax.scatter(start_coords[1], start_coords[0], c='blue', s=120, marker='o', edgecolors='white', label='Start')
        ax.scatter(goal_coords[1], goal_coords[0], c='magenta', s=120, marker='*', edgecolors='white', label='Goal')
        ax.set_xlim(-0.5, base_map.shape[1] - 0.5)
        ax.set_ylim(base_map.shape[0] - 0.5, -0.5)
        ax.set_aspect('equal')

    if path:
        rows = [p['position'][0] for p in path]
        cols = [p['position'][1] for p in path]
        ax4.plot(cols, rows, c='lime', lw=3, marker='.', markersize=8, label='Path')
        patches = []
        n_steps = len(path)
        step_indices = list(range(0, n_steps, max(1, n_steps//20)))
        for i in step_indices:
            r, c = path[i]['position']
            ang = path[i]['angle']
            fp = get_gripper_footprint_cached(int(round(r)), int(round(c)), int(round(ang)))
            for fr, fc in fp:
                if 0 <= fr < base_map.shape[0] and 0 <= fc < base_map.shape[1]:
                    patches.append(plt.Rectangle((fc-0.5, fr-0.5), 1, 1))
        if patches:
            pc = PatchCollection(patches, facecolor='cyan', edgecolor='blue', linewidth=0.3, alpha=0.15)
            ax4.add_collection(pc)
        final_state = path[-1]['object_state']
        for obj_id, (nr, nc) in final_state.items():
            ax4.scatter(nc, nr, marker='s', s=100, c='orange', edgecolors='black', label='Displaced objects' if obj_id == list(final_state.keys())[0] else "")
        push_actions = [p for p in path if p['action'] and 'push' in str(p['action'])]
        print(f"Path summary: {len(path)} steps, {len(push_actions)} pushes")
        action_text = f"Steps: {len(path)}\nPushes: {len(push_actions)}\nCost: {total_cost:.1f}"
        ax4.text(0.02, 0.98, action_text, transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(loc='upper right')
    ax4.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    output_filename = f"{data_dir}/{filename}_path_2d.png"
    plt.savefig(output_filename, bbox_inches='tight')


def plan_path_for_scene(base_map, region_map, start_coords, start_angle=0, log_dir="", filename=""):
    goal_coords = find_goal_location(base_map)
    path, total_cost = directional_contact_planner(start_coords, start_angle, goal_coords, base_map, region_map)

    visualise_path(base_map, region_map, path, start_coords, goal_coords, total_cost, data_dir=log_dir, filename=filename)

    return path, total_cost


def convert_astar_path_to_world(voxel_path, voxel_size=0.01, base_x=0.6, base_y=0.0,
                                start_coords=(97, 50), robot_init_position=None, target_position=None):

    start_row, start_col = start_coords

    z_start = robot_init_position[2]
    z_end = target_position[2]
    n = len(voxel_path)
    z_values = np.linspace(z_start, z_end, n)
    
    world_path = []

    for i, waypoint in enumerate(voxel_path):
        row, col = waypoint['position']
        yaw = waypoint['angle']
        
        x = base_x - 0.2 - (start_row - row) * voxel_size
        y = base_y + (col - start_col) * voxel_size
        z = z_values[i]

        world_path.append([x, y, z, yaw])
    
    world_path = np.asarray(world_path, dtype=float)
        
    return world_path


class AStar:
    def __init__(self, scene_name, base_grid, updated_grid, robot_init_position, target_position, log_dir, filename):
        self.scene_name = scene_name
        self.base_map = base_grid
        self.region_map = updated_grid
        self.robot_init_position = robot_init_position
        self.target_position = target_position
        self.log_dir = log_dir
        self.filename = filename


    def find_path(self, start_coords=(97, 50), start_angle=0):
        voxel_path, total_cost = plan_path_for_scene(
            base_map=self.base_map,
            region_map=self.region_map,
			start_coords=start_coords,
			start_angle=start_angle,
            log_dir=self.log_dir,
            filename=self.filename
		)
        world_path = convert_astar_path_to_world(voxel_path, start_coords=start_coords, robot_init_position=self.robot_init_position, target_position=self.target_position)
        world_path = world_path.tolist() 

        return world_path, total_cost

