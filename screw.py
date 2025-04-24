from typing import Dict, Any, Union
import numpy as np
import sapien.core as sapien

# Import ManiSkill2 base environment and registration utility
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.sapien_env import BaseEnv

# (Optional) import specific robot class for type hints
from mani_skill2.agents.robots.panda import Panda

@register_env("ScrewdriverRMA-v0", max_episode_steps=100)
class ScrewdriverRMAEnv(BaseEnv):
   
    # Specify supported robot types (ManiSkill2 will load the specified robot).
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda  # the robot agent (e.g., Panda arm)

    def __init__(self, *args, friction_range=(0.3, 1.0), **kwargs):
        """
        Initialize the ScrewdriverRMA environment.

        Args:
            friction_range (tuple): Range (min, max) for uniform random friction values for screwdriver and screw.
        """
        self.friction_range = friction_range
        super().__init__(*args, **kwargs)
        # Initialize placeholders for current friction values (for privileged info).
        self._current_screwdriver_friction = 0.0
        self._current_screw_friction = 0.0

    def _load_scene(self, options: Dict[str, Any]):
        """
        Load the scene with the screw and screwdriver objects and supporting structures (e.g., ground plane).
        Called once during environment initialization.
        """
        scene = self.scene

        # Create a ground plane (static actor) for stability.
        ground_material = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=np.array([1.0, 1.0, 0.01]), material=ground_material, 
                                   pose=sapien.Pose(p=np.array([0.0, 0.0, -0.01])))
        builder.add_box_visual(half_size=np.array([1.0, 1.0, 0.01]), 
                                pose=sapien.Pose(p=np.array([0.0, 0.0, -0.01])), color=np.array([0.5, 0.5, 0.5, 1.0]))
        self.ground = builder.build_static(name="ground")

        # Create the screw object (static, fixed in place).
        screw_radius = 0.01   # radius of the screw (collision shape)
        screw_length = 0.1    # length of the screw
        builder = scene.create_actor_builder()
        builder.add_capsule_collision(radius=screw_radius, half_length=screw_length / 2, material=None, density=1000.0)
        builder.add_capsule_visual(radius=screw_radius, half_length=screw_length / 2, color=np.array([0.8, 0.8, 0.2, 1.0]))
        self.screw = builder.build_static(name="screw")

        # Create the screwdriver object (dynamic, will be held by the robot).
        driver_radius = 0.005   # radius of the screwdriver rod
        driver_length = 0.2     # length of the screwdriver
        builder = scene.create_actor_builder()
        builder.add_capsule_collision(radius=driver_radius, half_length=driver_length / 2, material=None, density=500.0)
        builder.add_capsule_visual(radius=driver_radius, half_length=driver_length / 2, color=np.array([0.8, 0.1, 0.1, 1.0]))
        self.screwdriver = builder.build(name="screwdriver")
        # (Robot is loaded by BaseEnv according to SUPPORTED_ROBOTS.)

    def _initialize_episode(self, env_idx: Union[int, np.ndarray], options: Dict[str, Any]):
        """
        Reset the environment for a new episode. Randomize friction and set initial poses for the screwdriver and screw.
        """
        # Position the screw at a fixed location (e.g., in front of the robot).
        # Here we place it upright with its base near the ground.
        screw_pose = sapien.Pose(p=np.array([0.5, 0.0, 0.05]), q=np.array([0.0, 0.0, 0.0, 1.0]))
        self.screw.set_pose(screw_pose)

        # Place the screwdriver in the robot's gripper (at the tool center point).
        tcp_pose = self.agent.tcp.pose  # tool center point pose of the robot's end-effector
        self.screwdriver.set_pose(tcp_pose)

        # Randomize friction coefficients for screwdriver and screw.
        fric1 = float(np.random.uniform(self.friction_range[0], self.friction_range[1]))
        fric2 = float(np.random.uniform(self.friction_range[0], self.friction_range[1]))
        material1 = self.scene.create_physical_material(static_friction=fric1, dynamic_friction=fric1, restitution=0.0)
        material2 = self.scene.create_physical_material(static_friction=fric2, dynamic_friction=fric2, restitution=0.0)
        # Apply the new materials to all collision shapes of the screwdriver and screw.
        for shape in self.screwdriver.get_collision_shapes():
            shape.set_physical_material(material1)
        for shape in self.screw.get_collision_shapes():
            shape.set_physical_material(material2)
        # Store current friction values for privileged observation.
        self._current_screwdriver_friction = fric1
        self._current_screw_friction = fric2

        # (Robot joint state reset is handled by BaseEnv; we can also randomize robot joints here if needed.)

    def _get_obs_extra(self, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Construct task-specific observation components (to be merged with agent's own observation).
        """
        # Robot proprioceptive state (joint positions and velocities).
        qpos = self.agent.get_qpos()  # joint positions
        qvel = self.agent.get_qvel()  # joint velocities
        agent_state = np.concatenate([qpos, qvel], axis=0).astype(np.float32)

        # Screwdriver state (pose and velocities).
        screwdriver_pose = self.screwdriver.get_pose()
        screwdriver_pos = screwdriver_pose.p
        screwdriver_quat = screwdriver_pose.q  # quaternion (x, y, z, w)
        screwdriver_linvel = self.screwdriver.get_velocity()          # linear velocity (3,)
        screwdriver_angvel = self.screwdriver.get_angular_velocity()  # angular velocity (3,)
        screwdriver_state = np.concatenate(
            [screwdriver_pos, screwdriver_quat, screwdriver_linvel, screwdriver_angvel], axis=0
        ).astype(np.float32)

        # Screw state (pose and velocities).
        screw_pose = self.screw.get_pose()
        screw_pos = screw_pose.p
        screw_quat = screw_pose.q
        screw_linvel = np.zeros(3, dtype=np.float32)   # screw is static, no linear velocity
        screw_angvel = np.zeros(3, dtype=np.float32)   # screw is static, no angular velocity
        screw_state = np.concatenate([screw_pos, screw_quat, screw_linvel, screw_angvel], axis=0).astype(np.float32)

        # Privileged information (current friction coefficients).
        screwdriver_priv = np.array([self._current_screwdriver_friction], dtype=np.float32)
        screw_priv = np.array([self._current_screw_friction], dtype=np.float32)

        # Goal information: target position and orientation for alignment.
        # Use the screw's position and its axis (orientation) as the goal for the screwdriver.
        # Compute the screw's main axis (assuming local z-axis is the screw's axis).
        x, y, z, w = screw_quat  # (x, y, z, w) quaternion
        # Normalize quaternion to get proper rotation matrix.
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x_q, y_q, z_q, w_q = x/norm, y/norm, z/norm, w/norm
        else:
            x_q, y_q, z_q, w_q = x, y, z, w
        # Construct rotation matrix from quaternion.
        xx = x_q * x_q; yy = y_q * y_q; zz = z_q * z_q
        xy = x_q * y_q; xz = x_q * z_q; yz = y_q * z_q
        wx = w_q * x_q; wy = w_q * y_q; wz = w_q * z_q
        rot_matrix = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
        ], dtype=np.float32)
        screw_axis_world = rot_matrix[:, 2]  # screw's local z-axis in world coordinates
        goal_info = np.concatenate([screw_pos, screw_axis_world], axis=0).astype(np.float32)

        # Assemble observation dictionary.
        obs = {
            "task_id": np.array([0], dtype=np.int32),
            "agent_state": agent_state,
            "screwdriver_state": screwdriver_state,
            "screwdriver_priv_info": screwdriver_priv,
            "screw_state": screw_state,
            "screw_priv_info": screw_priv,
            "goal_info": goal_info
        }
        return obs

    def evaluate(self) -> Dict[str, float]:
        """
        Check task success and compute alignment errors.
        Returns:
            dict: {'success': bool, 'pos_error': float, 'ori_error': float}
        """
        # Get current poses.
        screwdriver_pose = self.screwdriver.get_pose()
        screw_pose = self.screw.get_pose()

        # Compute screwdriver tip position.
        # Assume screwdriver's local +Z axis is along its length.
        driver_half_length = 0.1  # half the screwdriver length
        x, y, z, w = screwdriver_pose.q
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x_q, y_q, z_q, w_q = x/norm, y/norm, z/norm, w/norm
        else:
            x_q, y_q, z_q, w_q = x, y, z, w
        xx = x_q * x_q; yy = y_q * y_q; zz = z_q * z_q
        xy = x_q * y_q; xz = x_q * z_q; yz = y_q * z_q
        wx = w_q * x_q; wy = w_q * y_q; wz = w_q * z_q
        rot_sd = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
        ], dtype=np.float32)
        screwdriver_axis = rot_sd[:, 2]  # screwdriver's local z-axis in world
        screwdriver_tip = screwdriver_pose.p + screwdriver_axis * driver_half_length

        # Compute screw head (top) position.
        screw_half_length = 0.05  # half the screw length
        x2, y2, z2, w2 = screw_pose.q
        norm2 = np.sqrt(x2*x2 + y2*y2 + z2*z2 + w2*w2)
        if norm2 > 0:
            x_q2, y_q2, z_q2, w_q2 = x2/norm2, y2/norm2, z2/norm2, w2/norm2
        else:
            x_q2, y_q2, z_q2, w_q2 = x2, y2, z2, w2
        XX = x_q2 * x_q2; YY = y_q2 * y_q2; ZZ = z_q2 * z_q2
        XY = x_q2 * y_q2; XZ = x_q2 * z_q2; YZ = y_q2 * z_q2
        WX = w_q2 * x_q2; WY = w_q2 * y_q2; WZ = w_q2 * z_q2
        rot_sc = np.array([
            [1 - 2*(YY + ZZ),     2*(XY - WZ),         2*(XZ + WY)],
            [2*(XY + WZ),         1 - 2*(XX + ZZ),     2*(YZ - WX)],
            [2*(XZ - WY),         2*(YZ + WX),         1 - 2*(XX + YY)]
        ], dtype=np.float32)
        screw_axis_vec = rot_sc[:, 2]  # screw's local z-axis in world
        screw_top = screw_pose.p + screw_axis_vec * screw_half_length

        # Calculate errors.
        pos_error = float(np.linalg.norm(screwdriver_tip - screw_top))
        cos_angle = float(np.dot(screwdriver_axis, screw_axis_vec))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        ori_error = float(np.arccos(cos_angle))

        success = bool(pos_error < 0.02 and ori_error < 0.1)
        return {"success": success, "pos_error": pos_error, "ori_error": ori_error}

    def step(self, action: np.ndarray):
        """
        Apply an action to the robot and advance the simulation by one step.

        Args:
            action (np.ndarray): Robot action (e.g., joint commands).

        Returns:
            observation (dict), reward (float), done (bool), info (dict)
        """
        # Apply the action to the robot and step the physics simulation.
        self.agent.set_action(action)
        self.scene.step()

        # Get the latest observation.
        obs = self.get_obs()  # BaseEnv provides get_obs() which uses _get_obs_extra

        # Compute evaluation metrics for reward and termination.
        eval_info = self.evaluate()
        # Dense reward: negative sum of position and orientation errors (smaller is better).
        reward = - (eval_info["pos_error"] + eval_info["ori_error"])
        # Episode termination: success or (time limit handled by wrapper via max_episode_steps).
        done = bool(eval_info["success"])
        # Info: include success flag and error metrics.
        info = {"success": eval_info["success"], "pos_error": eval_info["pos_error"], "ori_error": eval_info["ori_error"]}
        return obs, reward, done, info
