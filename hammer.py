import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from mani_skill2.envs import register_env
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.scene_builder import TableSceneBuilder


@register_env("HammeringRMA-v0", max_episode_steps=200)
class HammeringRMAEnv(BaseEnv):
    

    def __init__(self, friction_range=(0.3, 0.8), **kwargs):
        """Initialize the HammeringRMA environment."""
        # Range for random friction coefficients (min, max)
        self.friction_range = friction_range
        # Task identifier (single-task scenario, so this can be a constant)
        self.task_id = 0
        # Placeholders for friction values set each episode
        self.hammer_friction = None
        self.nail_friction = None
        # Define nail and hammer geometric parameters
        self.nail_length = 0.10  # length the nail can travel (meters)
        self.handle_length = 0.30  # hammer handle length
        self.handle_radius = 0.01  # hammer handle radius
        self.head_radius = 0.03  # hammer head radius
        # Call BaseEnv constructor (loads robot, etc.)
        super().__init__(**kwargs)

    def _load_agent(self, *args, **kwargs):
        """Load the robot agent into the scene with a suitable initial pose."""
        # Position the robot slightly behind the nail (e.g., negative X direction relative to nail).
        # Here we assume the robot is facing toward positive X direction on a table.
        return super()._load_agent(*args, pose=sapien.Pose(p=[-0.5, 0, 0]), **kwargs)

    def _load_scene(self, scene_config: dict):
        """Set up the scene by adding a table, the nail (with prismatic joint), and the hammer."""
        # Build a table scene (floor and table) with the table surface at z = 0.
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Create the nail as an articulation with a prismatic joint.
        scene = self.scene
        builder = scene.create_articulation_builder()
        # Base (fixed part, e.g., a small "wood block" on the table to hold the nail)
        base_link_builder = builder.create_link_builder()
        base_link_builder.set_name("nail_base")
        # Add a flat base (small box) to simulate the surface into which the nail is driven.
        base_size = np.array([0.1, 0.1, 0.01])  # half-size of the base (10cm x 10cm x 1cm)
        base_link_builder.add_box_collision(half_size=base_size, density=5000)
        base_link_builder.add_box_visual(half_size=base_size, color=[0.5, 0.3, 0.2, 1.0])
        # Nail link (dynamic part that slides into the base along a vertical axis)
        nail_link_builder = builder.create_link_builder(base_link_builder)
        nail_link_builder.set_name("nail")
        nail_link_builder.set_joint_name("nail_slide_joint")
        # Set up prismatic joint (slider) along the vertical axis.
        # We rotate the entire articulation's base so that the joint's axis (default X) aligns with global vertical.
        nail_link_builder.set_joint_properties(
            'prismatic',
            limits=[[0.0, self.nail_length]],  # nail can slide from 0 (fully in) to nail_length (out)
            friction=0.0, damping=0.0  # base friction set to 0 here, will randomize later
        )
        # Add the nail geometry (a thin capsule to represent the nail).
        # We align the capsule along the joint axis (which will be vertical after we orient the articulation).
        nail_link_builder.add_capsule_collision(radius=0.005, half_length=self.nail_length / 2,
                                                pose=sapien.Pose(p=[self.nail_length / 2, 0, 0]))
        nail_link_builder.add_capsule_visual(radius=0.005, half_length=self.nail_length / 2,
                                             color=[0.7, 0.7, 0.7, 1.0],
                                             pose=sapien.Pose(p=[self.nail_length / 2, 0, 0]))
        # Build the nail articulation
        self.nail_articulation = builder.build(fix_root_link=False)
        # Orient and position the nail's base: place the base on the table (z=0), and rotate so joint axis is vertical.
        # We rotate by -90 degrees about Y, so the articulation's base X-axis (joint axis) points up (world Z).
        base_pose = sapien.Pose(p=[0.0, 0.0, 0.001], q=euler2quat(0, -np.pi / 2, 0))
        self.nail_articulation.set_root_pose(base_pose)
        # Retrieve the prismatic joint and base link for later use
        self.nail_joint = None
        for joint in self.nail_articulation.get_joints():
            if joint.get_name() == "nail_slide_joint":
                self.nail_joint = joint
                break
        base_link = self.nail_articulation.get_links()[0]  # base link of the nail articulation
        # Increase friction of the base link against the table (to keep base fixed on the table)
        for shape in base_link.get_collision_shapes():
            material = shape.get_material()
            material.set_static_friction(1.0)
            material.set_dynamic_friction(1.0)

        # Create the hammer as a free (dynamic) actor.
        hammer_builder = scene.create_actor_builder()
        hammer_builder.set_name("hammer")
        # Add hammer handle (capsule)
        hammer_builder.add_capsule_collision(radius=self.handle_radius, half_length=self.handle_length / 2,
                                             pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        hammer_builder.add_capsule_visual(radius=self.handle_radius, half_length=self.handle_length / 2,
                                          color=[0.4, 0.4, 0.4, 1.0], pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        # Add hammer head (sphere at one end of the handle)
        head_offset = self.handle_length / 2 + self.head_radius  # distance from handle center to head center
        hammer_builder.add_sphere_collision(radius=self.head_radius, pose=sapien.Pose(p=[head_offset, 0, 0]))
        hammer_builder.add_sphere_visual(radius=self.head_radius, color=[0.2, 0.2, 0.2, 1.0],
                                         pose=sapien.Pose(p=[head_offset, 0, 0]))
        # Build the hammer actor
        self.hammer = hammer_builder.build(static=False)
        # (Note: The hammer will be attached to the robot hand in reset by a fixed drive.)

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode, randomizing friction and attaching the hammer to the robot."""
        # Reset the base environment (this resets the robot and simulation)
        obs, info = super().reset(seed=seed, options=options)
        # Randomize friction coefficients for hammer and nail
        self.hammer_friction = float(np.random.uniform(*self.friction_range))
        self.nail_friction = float(np.random.uniform(*self.friction_range))
        # Apply random friction to all hammer collision shapes
        for shape in self.hammer.get_collision_shapes():
            material = shape.get_material()
            material.set_static_friction(self.hammer_friction)
            material.set_dynamic_friction(self.hammer_friction)
        # Apply random friction (resistance) to the nail's prismatic joint
        if self.nail_joint is not None:
            self.nail_joint.set_friction(self.nail_friction)
        # Attach the hammer to the robot's end effector using a fixed drive (constraint)
        # Identify the robot's end-effector link (assuming a link name containing "hand")
        hand_link = None
        robot_links = self.agent.robot.get_links() if hasattr(self.agent, "robot") else []
        for link in robot_links:
            if "hand" in link.get_name().lower():
                hand_link = link
                break
        if hand_link is None:
            raise RuntimeError("Could not find a hand link on the robot to attach the hammer.")
        # Set the hammer's pose to align with the hand link before attaching
        self.hammer.set_pose(hand_link.get_pose())
        # Create a fixed drive (constraint) between the hammer and the hand link
        self.attach_drive = self.scene.create_drive(
            self.hammer, sapien.Pose(),
            hand_link, sapien.Pose(),
            stiffness=1e6, damping=1e5,
            rotational_stiffness=1e6, rotational_damping=1e5
        )
        # Optionally, ensure the robot's gripper is closed on the hammer (if applicable)
        try:
            qpos = self.agent.robot.get_qpos()
            # If the last two joints are gripper fingers (as in Panda), set them near zero (closed)
            if qpos.shape[0] >= 2:
                qpos[-2:] = 0.0
                self.agent.robot.set_qpos(qpos)
        except Exception:
            pass
        # Return the initial observation and info
        return self._get_obs(), info

    def _get_obs(self):
        """Construct the structured observation dictionary."""
        # Task ID (constant for this single task scenario)
        task_id = np.array([self.task_id], dtype=np.int32)
        # Agent (robot) state: joint positions and velocities
        if hasattr(self.agent, "robot"):
            qpos = self.agent.robot.get_qpos()
            qvel = self.agent.robot.get_qvel()
            agent_state = np.concatenate([qpos, qvel]).astype(np.float32)
        else:
            # Fallback: if self.agent is an articulation itself
            joints = self.agent.get_joints() if hasattr(self.agent, "get_joints") else []
            qpos = np.array([j.get_position() for j in joints], dtype=np.float32)
            qvel = np.array([j.get_velocity() for j in joints], dtype=np.float32)
            agent_state = np.concatenate([qpos, qvel]).astype(np.float32)
        # Hammer state: position (3) and orientation (4) as a quaternion
        hammer_pose = self.hammer.get_pose()  # sapien.Pose object
        hammer_pos = hammer_pose.p
        hammer_quat = hammer_pose.q
        hammer_state = np.concatenate([hammer_pos, hammer_quat]).astype(np.float32)
        # Nail state: current prismatic joint position (how far out the nail is) and velocity
        if self.nail_joint is not None:
            nail_position = np.array([self.nail_joint.get_position()], dtype=np.float32)  # scalar as array
            nail_velocity = np.array([self.nail_joint.get_velocity()], dtype=np.float32)
            nail_state = np.concatenate([nail_position, nail_velocity]).astype(np.float32)
        else:
            nail_state = np.zeros(2, dtype=np.float32)
        # Privileged info: friction coefficients
        hammer_priv_info = np.array([self.hammer_friction if self.hammer_friction is not None else 0.0],
                                    dtype=np.float32)
        nail_priv_info = np.array([self.nail_friction if self.nail_friction is not None else 0.0], dtype=np.float32)
        # Goal info: target nail depth (here 0.0 represents fully hammered in)
        goal_info = np.array([0.0], dtype=np.float32)
        # Assemble the observation dictionary
        return {
            "task_id": task_id,
            "agent_state": agent_state,
            "hammer_state": hammer_state,
            "hammer_priv_info": hammer_priv_info,
            "nail_state": nail_state,
            "nail_priv_info": nail_priv_info,
            "goal_info": goal_info
        }

    def step(self, action):
        """
        Apply an action to the environment and step the simulation.
        Returns (observation, reward, done, info).
        """
        # Use BaseEnv's step to handle physics stepping and robot control
        obs, reward, done, info = super().step(action)
        # (Optional) Compute a custom reward or modify reward if needed.
        # For example, one could use the distance between hammer and nail or nail depth as part of the reward.
        # Here we keep the default reward (if any) from BaseEnv or use a placeholder.
        return obs, reward, done, info

    def evaluate(self, **kwargs):
        """
        Evaluate the current state of the environment by measuring the hammer's proximity to the nail.

        Returns:
            dict: A dictionary containing the distance between the hammer head and the top of the nail.
        """
        # Compute world position of the hammer head center
        hammer_pose = self.hammer.get_pose()
        # The hammer head center in the hammer's local frame is at [handle_length/2 + head_radius, 0, 0]
        head_local_pos = np.array([self.handle_length / 2 + self.head_radius, 0, 0], dtype=np.float32)
        # Transform this local point to world coordinates
        hammer_rotation = hammer_pose.to_transformation_matrix()[:3, :3]
        hammer_head_pos = hammer_rotation.dot(head_local_pos) + hammer_pose.p
        # Compute world position of the top of the nail
        nail_link = self.nail_articulation.get_links()[1]  # the moving nail link (index 1, since index 0 is base)
        nail_link_pose = nail_link.get_pose()
        # The top of the nail in the nail link's local frame is at [nail_length, 0, 0] (since link X-axis is up)
        nail_top_local = np.array([self.nail_length, 0, 0], dtype=np.float32)
        nail_rotation = nail_link_pose.to_transformation_matrix()[:3, :3]
        nail_top_pos = nail_rotation.dot(nail_top_local) + nail_link_pose.p
        # Distance between hammer head center and nail top
        dist = np.linalg.norm(hammer_head_pos - nail_top_pos)
        return {"hammer_to_nail_distance": dist}
