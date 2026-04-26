import gc
import numpy as np
import torch
import trimesh
import copy
import os

# CuRobo
import curobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from scipy.spatial.transform import Rotation as _R

def _T_to_np_pos_rot(T, as_quat=True, w_first=False):
    position = T[:3, 3]
    rotation = _R.from_matrix(T[:3, :3])
    if as_quat:
        xyzw = rotation.as_quat()
        if w_first:
            return position, xyzw[[3, 0, 1, 2]]
        return position, xyzw
    return position, rotation.as_euler("xyz")

setup_curobo_logger("warn")

# NOTE: assumption is torch in, torch out


class CuroboMotionPlanner:
    def __init__(self, batch_size=2, dt=1. / 60, robot_path=None, world_file=None):

        # this is a minibatch size used internally
        self.batch_size = batch_size
        self.dt = dt

        assert robot_path is not None
        assert world_file is not None
        self.robot_path = robot_path
        self.world_file = world_file

        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        world_config_dir = os.path.join(
            _pkg_dir,
            '../curobo_content/configs/world')
        robot_config_dir = os.path.join(
            _pkg_dir,
            '../curobo_content/configs/robot')

        world_path_full = join_path(world_config_dir, self.world_file)
        print('world path full: ', world_path_full)
        self.world_cfg_dict = load_yaml(world_path_full)
        world_cfg = WorldConfig.from_dict(self.world_cfg_dict)
        world_cfg_list = []
        for _ in range(self.batch_size):
            world_cfg_list.append(world_cfg)

        self.world_cfg_list = world_cfg_list

        self.tensor_args = TensorDeviceType()
        # config_file = load_yaml(join_path(get_robot_configs_path(), self.robot_path))["robot_cfg"]
        config_file = load_yaml(os.path.join(robot_config_dir, self.robot_path))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(config_file, self.tensor_args)
        self.robot_cfg = robot_cfg
        self._init_motion_gen()
        self.j_names = robot_cfg.kinematics.cspace.joint_names
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=2,
            enable_finetune_trajopt=True,
            fail_on_invalid_query=True)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)
        self.n_dof = len(self.j_names)

    def _init_motion_gen(self):
        # Explicitly free the old MotionGen (holds large CUDA graphs) before allocating a new one
        if hasattr(self, 'motion_gen'):
            del self.motion_gen
            gc.collect()
            torch.cuda.empty_cache()

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg_list,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=self.dt,
            collision_cache={"obb": 4, "mesh": 1},
            collision_activation_distance=0.025,
            maximum_trajectory_dt=0.25,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.reset()

    def release_motion_gen(self):
        """Delete MotionGen and flush CUDA cache to free GPU memory (e.g. before FoundationPose)."""
        if hasattr(self, 'motion_gen'):
            del self.motion_gen
            gc.collect()           # break CUDA-graph reference cycles before flushing
            torch.cuda.empty_cache()

    def ensure_motion_gen(self):
        """Reinitialize MotionGen if it was previously released."""
        if not hasattr(self, 'motion_gen'):
            self._init_motion_gen()

    def fk(self, q, curobo_type=False):
        assert len(q.shape) == 2, q.shape
        state = self.motion_gen.compute_kinematics(
            curobo.types.state.JointState.from_position(q)
        )

        pose = Pose(
            state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
        )

        if curobo_type:
            return pose

        return pose.get_matrix()

    def lift(self, current_q: torch.Tensor, height=0.25):
        # B x N_DOF
        assert len(current_q.shape) == 2
        assert current_q.shape[1] == self.n_dof
        # B x 4 x 4
        goal_pose = self.fk(current_q)
        goal_pose[:, 2, 3] += height

        return self.plan_trajectory_to_pose_from_q(goal_pose, current_q)

    def plan_trajectory_to_pose_from_q(self, goal_pose: torch.Tensor, q_start: torch.Tensor):
        B = q_start.shape[0]
        assert q_start.shape == (B, self.n_dof), q_start.shape
        assert goal_pose.shape == (B, 4, 4), goal_pose.shape

        goal_pose_curobo = curobo.types.math.Pose.from_matrix(goal_pose)
        start_state_curobo = curobo.types.state.JointState.from_position(q_start)

        if B != self.batch_size:
            assert B == 1, B
            goal_pose_curobo = goal_pose_curobo.repeat_seeds(self.batch_size)
            start_state_curobo = start_state_curobo.repeat_seeds(self.batch_size)

        return self._plan_trajectory_to_pose_from_state(goal_pose_curobo, start_state_curobo)

    def _plan_trajectory_to_pose_from_state(self, goal_pose: curobo.types.math.Pose, start_state: curobo.types.state.JointState):
        # B x T x N_DOF
        traj_pos, traj_vel, success = self._plan_trajectory_to_pose_from_q(goal_pose, start_state)
        traj_found = False
        success_id = None
        for i in range(len(success)):
            if success[i]:
                print('planning success: trajectory found')
                traj_found = True
                success_id = i
                break

        if not traj_found:
            print("ooops: not a single trajectory not found!")
            # NOTE: this should return why is it not found: env collision, joint limits etc.
            return None, None, None

        # return only one trajectory
        traj_pos = traj_pos[success_id]
        traj_vel = traj_vel[success_id]

        # T x N_DOF
        assert traj_pos.shape == traj_vel.shape
        assert len(traj_pos.shape) == 2, traj_pos.shape
        assert traj_pos.shape[1] == self.n_dof

        return traj_pos, traj_vel, success_id

    def plan_trajectory_to_pose_from_q_all(self, goal_pose: torch.Tensor, q_start: torch.Tensor):
        """Plan trajectories for a batch and return ALL successful results.

        Args:
            goal_pose: (B, 4, 4) target poses where B <= batch_size
            q_start: (B, N_DOF) starting joint configs where B <= batch_size

        Returns:
            results: list of (traj_pos, traj_vel, batch_index) for each successful plan
                     traj_pos/vel are (T, N_DOF). Returns empty list if none succeed.
        """
        B = q_start.shape[0]
        assert q_start.shape == (B, self.n_dof), q_start.shape
        assert goal_pose.shape == (B, 4, 4), goal_pose.shape
        assert B <= self.batch_size, f"Input batch size {B} exceeds planner batch size {self.batch_size}"

        # Pad to batch_size by repeating elements
        # E.g., B=2, batch_size=16: repeat 8 times each
        # E.g., B=7, batch_size=16: repeat all twice (14) + first 2 = 16
        if B < self.batch_size:
            repeats = self.batch_size // B
            remainder = self.batch_size % B
            if remainder > 0:
                goal_pose = torch.cat([goal_pose.repeat(repeats, 1, 1), goal_pose[:remainder].clone()], dim=0)
                q_start = torch.cat([q_start.repeat(repeats, 1), q_start[:remainder].clone()], dim=0)
            else:
                goal_pose = goal_pose.repeat(repeats, 1, 1)
                q_start = q_start.repeat(repeats, 1)

        goal_pose_curobo = curobo.types.math.Pose.from_matrix(goal_pose)
        start_state_curobo = curobo.types.state.JointState.from_position(q_start)

        traj_pos, traj_vel, success = self._plan_trajectory_to_pose_from_q(
            goal_pose_curobo, start_state_curobo)

        # Map padded indices back to original indices and collect unique successes
        results = []
        seen_original_indices = set()
        for i in range(len(success)):
            if success[i]:
                original_idx = i % B  # Map back to original index
                if original_idx not in seen_original_indices:
                    seen_original_indices.add(original_idx)
                    results.append((traj_pos[i], traj_vel[i], original_idx))

        print(f"Batch planning: {len(results)}/{B} trajectories succeeded")
        return results

    def plan_trajectory_home(self, current_q: torch.Tensor):
        assert len(current_q.shape) == 2
        assert current_q.shape[1] == self.n_dof
        home_pose = self.fk(torch.zeros(1, 7).cuda())
        return self.plan_trajectory_to_pose_from_q(home_pose, current_q)

    def check_can_plan_home(self, q: torch.Tensor) -> bool:
        """Check if we can plan home from given joint position without executing.

        Args:
            q: Joint positions tensor of shape (1, n_dof) or (n_dof,)

        Returns:
            bool: True if planning home succeeds, False otherwise
        """
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        assert len(q.shape) == 2
        assert q.shape[1] == self.n_dof

        try:
            traj_pos, traj_vel, success_id = self.plan_trajectory_home(q)
            return traj_pos is not None
        except Exception:
            return False

    def check_can_plan_home_batch(self, q_batch: torch.Tensor) -> list:
        """Check if we can plan home from multiple joint positions (batched).

        Args:
            q_batch: Joint positions tensor of shape (B, n_dof) where B <= batch_size

        Returns:
            list[bool]: List of length B, True if planning home succeeds for each
        """
        assert len(q_batch.shape) == 2
        assert q_batch.shape[1] == self.n_dof
        B = q_batch.shape[0]
        assert B <= self.batch_size, f"Input batch {B} exceeds planner batch size {self.batch_size}"

        # Home pose is the same for all
        home_pose = self.fk(torch.zeros(1, 7).cuda())
        home_pose_batch = home_pose.repeat(B, 1, 1)

        # plan_trajectory_to_pose_from_q_all handles padding internally
        results = self.plan_trajectory_to_pose_from_q_all(home_pose_batch, q_batch)

        # Build success array
        success_indices = set(r[2] for r in results)
        can_plan_home = [i in success_indices for i in range(B)]

        return can_plan_home

    def _plan_trajectory_to_pose_from_q(
            self,
            goal_pose: curobo.types.math.Pose,
            start_state: curobo.types.state.JointState):
        print('planning..')
        result = self.motion_gen.plan_batch_env(start_state, goal_pose, self.plan_config.clone())
        if result.interpolated_plan is None:
            print('planning failed: no paths returned')
            return [], [], [False] * goal_pose.position.shape[0]
        trajs = result.get_paths()
        traj_pos = []
        traj_vel = []
        for s in range(len(trajs)):
            # Get controlled joint state from input joint state.
            # This is used to get the joint state for only joints that are optimization variables.
            # This also re-orders the joints to match the order of optimization variables.
            traj_pos.append(self.motion_gen.get_active_js(trajs[s]).position)
            traj_vel.append(self.motion_gen.get_active_js(trajs[s]).velocity)
        return traj_pos, traj_vel, result.success

    def check_configs_collision_batch(self, q_batch: torch.Tensor) -> torch.Tensor:
        """
        Check if joint configurations are in collision (world obstacles + self-collision).

        Uses MotionGen.check_constraints() which runs the IK rollout constraint checker,
        covering both world collision and self-collision.

        Args:
            q_batch : (B, N_DOF) joint configs on any device

        Returns:
            in_collision : (B,) bool tensor — True = in collision
        """
        q = q_batch.to(self.tensor_args.device)
        state = curobo.types.state.JointState.from_position(q)
        metrics = self.motion_gen.check_constraints(state)
        # feasible is True when config satisfies all constraints (no collision)
        in_collision = ~metrics.feasible.squeeze(-1)
        return in_collision  # (B,) bool on tensor_args.device

    def plan_trajectory_to_js_batch(self, goal_qs: torch.Tensor, q_start: torch.Tensor):
        """Batch JS→JS planning: one or B start configs to B different goal configs.

        CuRobo's plan_single_js is hardcoded to batch_size=1.  This method replicates
        its no-graph trajopt path with the view() fixed for batch_size=B, then calls
        _solve_trajopt_from_solve_state directly.

        Args:
            goal_qs: (B, N_DOF) target joint configs
            q_start: (1, N_DOF) or (N_DOF,) — broadcast to all B
                     (B, N_DOF) — different start per goal (e.g. approach→grasp)

        Returns:
            list of (traj_pos, traj_vel, batch_idx) — one per successful plan
        """
        from curobo.rollout.rollout_base import Goal
        from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType

        if len(goal_qs.shape) == 1:  goal_qs = goal_qs.unsqueeze(0)
        if len(q_start.shape) == 1:  q_start = q_start.unsqueeze(0)

        B   = goal_qs.shape[0]
        dev = self.tensor_args.device

        goal_qs_d = goal_qs.contiguous().to(dev)
        # Support B different starts (approach→grasp) or one start broadcast to all B
        if q_start.shape[0] == B:
            q_starts = q_start.contiguous().to(dev)
        else:
            q_starts = q_start.expand(B, -1).contiguous().to(dev)

        start_state = curobo.types.state.JointState.from_position(q_starts)
        goal_state  = curobo.types.state.JointState.from_position(goal_qs_d)

        num_seeds = self.motion_gen.js_trajopt_solver.num_seeds
        h         = self.motion_gen.js_trajopt_solver.action_horizon

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            batch_size=B,
            n_envs=1,
            n_goalset=1,
            num_trajopt_seeds=num_seeds,
            num_graph_seeds=num_seeds,
        )

        traj_result = None
        try:
            # Build linear seeds: repeat each (start, goal) pair by num_seeds so
            # get_seed_set generates (num_seeds * B, h, dof).
            seed_goal = Goal(
                current_state=start_state.repeat_seeds(num_seeds),
                goal_state=goal_state.repeat_seeds(num_seeds),
            )
            seed_traj = self.motion_gen.js_trajopt_solver.get_seed_set(
                seed_goal, None, num_seeds=1, batch_mode=False, seed_success=None,
            )
            # Fix the view that plan_single_js hardcodes as (num_seeds*1, 1, h, dof):
            # reshape (num_seeds*B, h, dof) → (num_seeds, B, h, dof)
            seed_traj = seed_traj.view(num_seeds, B, h, self.n_dof).contiguous().clone()

            goal = Goal(current_state=start_state, goal_state=goal_state)

            traj_result = self.motion_gen._solve_trajopt_from_solve_state(
                goal, solve_state, seed_traj,
                num_seeds_override=num_seeds,
                newton_iters=self.motion_gen.js_trajopt_solver.newton_iters,
                return_all_solutions=False,
                trajopt_instance=self.motion_gen.js_trajopt_solver,
            )

            # Optional finetune (mirrors plan_single_js finetune loop)
            plan_config = self.plan_config.clone()
            if plan_config.enable_finetune_trajopt and traj_result.success.any():
                seed_ft = traj_result.raw_action.clone()
                opt_dt  = traj_result.optimized_dt
                opt_dt  = torch.min(opt_dt[traj_result.success])
                for k in range(plan_config.finetune_attempts):
                    scaled_dt = torch.clamp(
                        opt_dt * plan_config.finetune_js_dt_scale
                               * (plan_config.finetune_dt_decay ** k),
                        self.motion_gen.js_trajopt_solver.minimum_trajectory_dt,
                    )
                    if self.motion_gen.optimize_dt:
                        self.motion_gen.finetune_js_trajopt_solver.update_solver_dt(
                            scaled_dt.item())
                    ft_result = self.motion_gen._solve_trajopt_from_solve_state(
                        goal, solve_state, seed_ft,
                        trajopt_instance=self.motion_gen.finetune_js_trajopt_solver,
                        num_seeds_override=num_seeds,
                        newton_iters=None,
                        return_all_solutions=False,
                    )
                    if ft_result.success.any() or not self.motion_gen.optimize_dt:
                        traj_result = ft_result
                        break
                    seed_ft = ft_result.optimized_seeds.detach().clone()

        except Exception as e:
            print(f'plan_trajectory_to_js_batch: solve failed: {e}')
            return []

        if traj_result is None or traj_result.success is None or not traj_result.success.any():
            print('plan_trajectory_to_js_batch: no solutions')
            return []

        # Parse TrajOptResult — for BATCH solve type:
        #   success                : (B,) bool
        #   interpolated_solution  : JointState, position shape (B, T, dof)
        #   path_buffer_last_tstep : (B,) int — valid endpoint per trajectory
        success   = traj_result.success
        path_ends = traj_result.path_buffer_last_tstep
        interp    = traj_result.interpolated_solution

        results = []
        for i in range(min(B, len(success))):
            if not success[i]:
                continue
            try:
                end    = int(path_ends[i]) if (path_ends is not None and i < len(path_ends)) else None
                traj_i = interp[i:i+1]          # JointState slice: position (1, T, dof)
                js_act = self.motion_gen.get_active_js(traj_i)
                pos    = js_act.position[:, :end] if end else js_act.position  # (1, T, dof)
                vel    = js_act.velocity[:, :end] if end else js_act.velocity
                if pos.dim() == 3:               # squeeze batch dim → (T, dof)
                    pos = pos.squeeze(0)
                    vel = vel.squeeze(0)
                results.append((pos, vel, i))
            except Exception as e:
                print(f'plan_trajectory_to_js_batch: result[{i}] parse error: {e}')

        print(f'plan_trajectory_to_js_batch: {len(results)}/{B} succeeded')
        return results

    def plan_trajectory_to_js(
        self,
        goal_q:  torch.Tensor,
        q_start: torch.Tensor,
    ):
        """
        Plan a collision-free trajectory from q_start to a specific joint configuration
        (JS target, not pose target).  Uses CuRobo's plan_single_js.

        Args:
            goal_q  : (1, N_DOF) or (N_DOF,) target joint config
            q_start : (1, N_DOF) or (N_DOF,) start  joint config

        Returns:
            traj_pos : (T, N_DOF) or None if planning fails
            traj_vel : (T, N_DOF) or None
        """
        if len(goal_q.shape)  == 1: goal_q  = goal_q.unsqueeze(0)
        if len(q_start.shape) == 1: q_start = q_start.unsqueeze(0)

        dev = self.tensor_args.device
        goal_state  = curobo.types.state.JointState.from_position(goal_q.to(dev))
        start_state = curobo.types.state.JointState.from_position(q_start.to(dev))

        try:
            result = self.motion_gen.plan_single_js(
                start_state, goal_state, self.plan_config.clone())
        except Exception as e:
            print(f'plan_trajectory_to_js exception: {e}')
            return None, None

        if result.success is None or not result.success.any():
            print('plan_trajectory_to_js: no solution found')
            return None, None

        traj = result.get_interpolated_plan()
        if traj is None:
            return None, None

        traj_pos = self.motion_gen.get_active_js(traj).position   # (T, N_DOF)
        traj_vel = self.motion_gen.get_active_js(traj).velocity
        return traj_pos, traj_vel

    def score_trajectory_joint_travel(self, traj_pos: torch.Tensor) -> float:
        """Score a trajectory by total joint-space travel (lower travel = better).

        Returns the negative sum of per-step L1 joint displacements so that
        higher scores mean less joint motion (compatible with sort(..., reverse=True)).
        """
        if isinstance(traj_pos, np.ndarray):
            traj_pos = torch.from_numpy(traj_pos)
        traj_pos = traj_pos.float().to(self.tensor_args.device)
        if traj_pos.dim() == 1:
            traj_pos = traj_pos.unsqueeze(0)

        if traj_pos.shape[0] < 2:
            return 0.0

        total_travel = (traj_pos[1:] - traj_pos[:-1]).abs().sum().item()
        return -total_travel

    def get_link_transforms(self, q: torch.Tensor):
        """Compute FK for all tracked links at joint config q.

        Args:
            q: (N_DOF,) or (1, N_DOF) joint config

        Returns:
            list of (link_name, T_4x4_np) — one entry per tracked link, in world frame
        """
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        q = q.to(self.tensor_args.device)

        state = self.kin_model.get_state(q)
        # links_position: (1, n_links, 3), links_quaternion: (1, n_links, 4) wxyz
        pos  = state.links_position[0].cpu().numpy()   # (n_links, 3)
        quat = state.links_quaternion[0].cpu().numpy()  # (n_links, 4) wxyz

        from scipy.spatial.transform import Rotation
        results = []
        for j, name in enumerate(self.kin_model.link_names):
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = pos[j]
            # scipy uses xyzw; curobo wxyz → xyzw
            xyzw = np.array([quat[j, 1], quat[j, 2], quat[j, 3], quat[j, 0]])
            T[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
            results.append((name, T))
        return results

    def get_joint_names(self):
        return [_ for _ in self.j_names]

    def update_world_model_with_trimesh(self, obj_mesh: trimesh.Trimesh, scale: float, T: torch.Tensor):
        assert T.shape == (4, 4), T.shape

        faces = obj_mesh.faces
        vertices = obj_mesh.vertices
        print(f"loading mesh with {faces} faces and {vertices}")

        xyz, wxyz = _T_to_np_pos_rot(T, as_quat=True, w_first=True)
        print("all previous meshes are removed")
        world_cfg_dict = copy.deepcopy(self.world_cfg_dict)
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
        # assumes only 1
        xyz = xyz.squeeze()
        wxyz = wxyz.squeeze()

        print('adding the obstacle (1 only)')
        obstacle = Mesh(
            name="moving_object",
            pose=[xyz[0], xyz[1], xyz[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3]],
            faces=faces,
            vertices=vertices,
            scale=[scale, scale, scale],
        )
        world_cfg.add_obstacle(obstacle)

        world_cfg_list = []
        for _ in range(self.batch_size):
            world_cfg_list.append(world_cfg)

        self.world_cfg_list = world_cfg_list
        self._init_motion_gen()
