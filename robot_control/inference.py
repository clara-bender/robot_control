from collections import deque
import threading
from threading import Thread
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from cv_bridge import CvBridge
import numpy as np

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        # self.FPS_INFER = 60.0 # still finding upper limit on this
        # self.DT = 1.0 / self.FPS_INFER
        # self.CONTROL_HZ = 150.0 # multiple of 10
        self.PREDICTION_HORIZON = 20
        self.MIN_EXECUTION_HORIZON = 10
        self.ROBOT_DOF = 7
        self.DELAY_INIT = 5
        self.BUFFER_SIZE = 5
        self.initialize_started = False
        
        # Load the trained policy
        config = _config.get_config("pi05_xarm_finetune")
        checkpoint_dir = download.maybe_download(
            "/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
        )
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # Master frequency subscription
        self.frequency_sub = self.create_subscription(Float32, '/master_frequency', self.frequency_callback, 10)

        # Camera Subscriptions
        self.wrist_sub = self.create_subscription(Image, '/camera_image/wrist', self.wrist_camera_callback, 10)
        self.tripod_sub = self.create_subscription(Image, '/camera_image/tripod', self.tripod_camera_callback, 10)

        # Robot state subscription
        self.servo_state_sub = self.create_subscription(Float32MultiArray, '/servo_state', self.servo_state_callback, 10)
        self.gripper_state_sub = self.create_subscription(Int32, '/gripper_state', self.gripper_state_callback, 10)

        # Publisher to send commands
        self.gripper_pub = self.create_publisher(Int32, '/cmd_auto_gripper', 10)
        self.servo_pub = self.create_publisher(Float32MultiArray, '/cmd_auto_servo', 10)


        # Initializations
        self.mutex = threading.Lock()
        self.condition_variable = threading.Condition(self.mutex)
        self.t = 0
        self.time_since_last_inference = 0
        self.wrist_camera = None
        self.tripod_camera = None
        self.servo_state = None
        self.gripper_state = 840.0
        self.initialized = False
        self.observation_curr = None
        self.action_curr = None

        print('I exist')
    
    def frequency_callback(self, msg):
        self.MASTER_HZ = msg.data
        self.DT = 1.0 / self.MASTER_HZ
        self.get_logger().info(f"Received Master Frequency: {self.MASTER_HZ} Hz")

    def servo_state_callback(self, msg):
        servo_state = np.array(msg.data, dtype=np.float32)
        pose = servo_state.tolist()
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360

        if not self.initialize_started:
            self.try_initialize(pose)
            return
        if not self.initialized:
            return
    
        observation = self.get_observation(pose)
        command = self.get_action(observation)
        cmd_joint_pose = command[:6].copy()
        cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180
        state = np.array(pose, dtype=np.float32)
        self.interpolate_action(state, cmd_joint_pose)
        cmd_gripper_pose = int((command[6]) * -860 + 850) # unnormalize the gripper action
        
        gripper_msg = Int32(data=cmd_gripper_pose)
        self.gripper_pub.publish(gripper_msg)
        self.get_logger().info(f"Gripper: {gripper_msg.data}")

    def gripper_state_callback(self, msg):
        self.gripper_state = float(msg.data)
        self.get_logger().info(f"Gripper state updated: {self.gripper_state}")

    def wrist_camera_callback(self, msg):
        bridge = CvBridge()
        self.wrist_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # self.observation["observation/wrist_image_left"] = self.wrist_camera

    def tripod_camera_callback(self, msg):
        bridge = CvBridge()
        self.tripod_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def try_initialize(self,pose):
        if self.wrist_camera is not None and self.tripod_camera is not None:
            self.initialize_started = True
            self.get_logger().info("Initial data received. Running first inference...")

            self.observation_curr = self.get_observation(pose)
            self.action_curr = np.array(
                self.policy.infer(self.observation_curr)["actions"],
                dtype=np.float32
            )

            self.inference_loop_thread = Thread(target=self.inference_loop)
            self.inference_loop_thread.start()

            self.initialized = True
            self.get_logger().info("Inference loop started.")
        else:
            self.get_logger().info("Waiting for camera images...")

    def get_observation(self,pose):
        
        angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
        state = np.array(pose[:3] + angles_rad, dtype=np.float32)

        g_p = self.gripper_state
        self.get_logger().info(f"Current gripper state: {g_p}")
        g_p = np.array((g_p - 850) / -860)

        observation = {
            "observation/exterior_image_1_left": self.tripod_camera,
            "observation/exterior_image_2_left": np.zeros_like(self.tripod_camera),  # Placeholder for second exterior image
            "observation/wrist_image_left": self.wrist_camera,
            "observation/gripper_position": g_p,
            "observation/joint_position": state,
            "prompt": "Pick up the bag and place it on the blue x",
        }

        return observation
    
    def get_action(self, observation_next):
        with self.condition_variable:
            self.get_logger().info("Getting action")
            self.t += 1

            self.observation_curr = observation_next
            self.condition_variable.notify()

            action = self.action_curr[self.t - 1, :].copy()

        return action
    
    def guided_inference(self, observation, action_prev, delay, time_since_last_inference):
        H = self.PREDICTION_HORIZON
        i = np.arange(delay, H - time_since_last_inference)
        c = (H - time_since_last_inference - i) / (H - time_since_last_inference - delay + 1)

        W = np.ones(H)
        W[0:delay] = 1.0
        W[delay:H - time_since_last_inference] = c * (np.exp(c) - 1) / (np.exp(1) - 1)
        W[H - time_since_last_inference:] = 0.0

        T, robot_dof = action_prev.shape
        if T < H:
            action_prev = np.pad(action_prev, ((0, H - T), (0, 0)), mode='constant')

        v_pi = np.array(self.policy.infer(observation)["actions"])
        v_pi = v_pi[:H, :self.ROBOT_DOF]  # ensure correct shape

        A = action_prev.copy()
        action_estimate = A*W[:,None] + v_pi*(1-W[:, None])

        return action_estimate[:H, :self.ROBOT_DOF]
    
    def interpolate_action(self, state, goal):
        command = goal.copy()
        command[3] = (command[3]+ 180) % 360 -180
        command[5] = (command[5]+ 180) % 360 -180

        self.cmd_servo = command
        servo_msg = Float32MultiArray(data=command.tolist())
        self.servo_pub.publish(servo_msg)
        self.get_logger().info(f"Servo: {servo_msg.data}")

    def inference_loop(self):
        while True:
            Q = deque([self.DELAY_INIT], maxlen=self.BUFFER_SIZE)

            with self.condition_variable:
                    while self.t < self.MIN_EXECUTION_HORIZON:
                        self.get_logger().info(f"Waiting for {self.MIN_EXECUTION_HORIZON - self.t} more steps before another inference...")
                        self.condition_variable.wait()

                    time_since_last_inference = self.t
                    # Remove actions that have already been executed
                    action_prev = self.action_curr[
                        time_since_last_inference:self.PREDICTION_HORIZON
                    ].copy() 

                    delay = max(Q)
                    # print("Delay: ", delay)
                    obs = self.observation_curr.copy()

            # ---- lock released ----
            action_new = self.guided_inference(
                obs,
                action_prev,
                delay,
                time_since_last_inference
            )

            self.action_curr[:action_new.shape[0], :] = action_new
            self.t = self.t - time_since_last_inference
            Q.append(self.t)

    def action_loop(self):
        while True:
            self.get_logger().info("Action loop going.")

            t0 = time.perf_counter()

            observation = self.get_observation()
            self.get_logger().info("Observation obtained, running inference...")
            command = self.get_action(observation)

            cmd_joint_pose = command[:6].copy()
            cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

            pose = self.servo_state.tolist()
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            state = np.array(pose, dtype=np.float32)

            # execute smooth motion to target via interpolation
            self.interpolate_action(state, cmd_joint_pose)
            cmd_gripper_pose = int((command[6]) * -860 + 850) # unnormalize the gripper action
            
            gripper_msg = Int32(data=cmd_gripper_pose)
            self.gripper_pub.publish(gripper_msg)
            self.get_logger().info(f"Gripper: {gripper_msg.data}")

            time_left = self.DT - (time.perf_counter() - t0)
            self.get_logger().info(f"Time left: {time_left:.3f} seconds")
            time.sleep(max(time_left, 0))
            self.get_logger().info("Sleep time over")

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()