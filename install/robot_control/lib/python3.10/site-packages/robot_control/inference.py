import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Int32
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xarm.wrapper import XArmAPI
import time
import threading
from collections import deque
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # =========================
        # User inputs
        # =========================

        self.XARM_IP = '192.168.1.222'

        # Inference
        self.CONFIG_NAME = "pi05_xarm_finetune"
        self.CHECKPOINT_FOLDER = "/home/admin/imdying/src/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
        self.FPS = 80.0 # still finding upper limit on this
        self.DT = 1.0 / self.FPS
        self.CONTROL_HZ = 100.0 # multiple of 10
        self.PREDICTION_HORIZON = 20
        self.MIN_EXECUTION_HORIZON = 10
        self.DELAY_INIT = 5
        self.BUFFER_SIZE = 5
        self.ROBOT_DOF = 7
        self.FPS_COLLECT = 10

        # Collection
        self.REPO_NAME = "clara/new"
        self.TASK_DESCRIPTION = "Pick up the bag and place it in the center of the workspace."

        # =========================
        # Variables
        # =========================
        self.t = 0
        self.start = False
        self.start_inference = False
        self.start_correction = False
        self.start_collection = False
        self.manual_mode = False
        self.infer_thread = None
        self.exec_thread = None
        self.publish_state_thread = None
        self.execute = False
        self.discard_data = False
        self.frames_recorded = 0
        self.failure = False
        self.prev_data = None
        mutex = threading.Lock()
        self.condition_variable = threading.Condition(mutex)
        self.collect_loop = None
        self.publish_observation_thread = None

        self.manual_gripper = None
        self.manual_servo = None

        # =========================
        # Policy Setup
        # =========================
        config = _config.get_config(self.CONFIG_NAME)
        checkpoint_dir = download.maybe_download(self.CHECKPOINT_FOLDER)
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # =========================
        # XArm Setup
        # =========================
        self.arm = XArmAPI(self.XARM_IP)

        if self.arm.get_state() != 0:
            self.arm.clean_error()
            time.sleep(0.5)

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_gripper_mode(0)

        self.go_home()

        # =========================
        # RealSense Camera Setup
        # =========================
        self.wrist_camera = None
        self.exterior_camera = None
        # self.servo_state = None
        # self.gripper_state = None
        # ctx = rs.context()
        # devices = ctx.query_devices()

        # if len(devices) < 2:
        #     raise RuntimeError("Need at least two RealSense cameras connected")

        # serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        # print("Found cameras:", serials)

        # self.pipelines = []
        # self.configs = []

        # for serial in serials:
        #     pipeline = rs.pipeline()
        #     config_rs = rs.config()
        #     config_rs.enable_device(serial)
        #     config_rs.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
        #     pipeline.start(config_rs)
        #     self.pipelines.append(pipeline)
        #     self.configs.append(config_rs)

        # =========================
        # Dataset Setup
        # =========================

        # if self.start_collection:
        self.dataset_path = HF_LEROBOT_HOME / self.REPO_NAME

        if self.dataset_path.exists(): 
            self.dataset = LeRobotDataset(
                root=self.dataset_path,
                repo_id=self.REPO_NAME,
            )
            print("Adding to existing dataset, waiting for signal.")
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.REPO_NAME,
                robot_type="xarm",
                fps=self.FPS_COLLECT,
                features={
                    "exterior_image_1_left": {
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "exterior_image_2_left": { # this one is not used, put it as zeros or something
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image_left": {
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "joint_position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["joint_position"],
                    },
                    "gripper_position": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["gripper_position"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                        "names": ["actions"],
                    },
                },
            )

        # =========================
        # Subscriptions
        # =========================
        self.start_sub = self.create_subscription(Bool, 'start_button', self.start_callback, 10)
        self.wrist_sub = self.create_subscription(Image, '/camera_image/wrist', self.wrist_camera_callback, 10)
        self.exterior_sub = self.create_subscription(Image, '/camera_image/tripod', self.exterior_camera_callback, 10)
        # self.servo_sub = self.create_subscription(Float32MultiArray, '/servo_state', self.servo_callback, 10)
        # self.gripper_sub = self.create_subscription(Int32, '/gripper_state', self.gripper_callback, 10)
        
        # Subscriber: gui.py execution selection
        self.execution_sub = self.create_subscription(Bool, '/execution_selection',self.execution_callback,10)

        # Subscriber: correction.py, "manual" commands
        self.manual_gripper_sub = self.create_subscription(Int32,'/cmd_manual_gripper',self.manual_gripper_callback,10)
        self.manual_servo_sub = self.create_subscription(Float32MultiArray,'/cmd_manual_servo',self.manual_servo_callback,10)

        # Subscriber: correction.py, mode toggle (left and right button), True for manual, False for auto
        self.mode_sub = self.create_subscription(Bool,'/manual_mode',self.mode_callback,10)
        # self.auto_servo_pub = self.create_publisher(Float32MultiArray, '/cmd_auto_servo', 10)
        # self.auto_gripper_pub = self.create_publisher(Int32, '/cmd_auto_gripper', 10)
        # Publisher: Current robot state
        self.servo_state_pub = self.create_publisher(Float32MultiArray, '/servo_state', 10)
        self.gripper_state_pub = self.create_publisher(Int32, '/gripper_state', 10)

        # Subscribers: gui.py, save or discard data
        self.save_sub = self.create_subscription(Bool, '/save_button', self.save_callback, 10)

        # Subscribers: gui.py, failure or success signal
        self.failure_success_sub = self.create_subscription(Bool, '/failure_success_button', self.failure_success_callback, 10)
        

        # self.timer_state = self.create_timer(1/40, self.publish_state)  # 100Hz
        

        print("Ready to go!")

    # =========================
    # Start/Stop
    # =========================
    def start_callback(self, msg):
        self.start = msg.data
        if self.start and not self.manual_mode:
            self.start_inference = True
            self.start_infer()
        elif self.start and self.manual_mode:
            self.observation_curr = self.get_observation()
            self.start_correction = True
            # if self.publish_observation_thread is None:
            #     self.publish_observation_thread = threading.Thread(target=self.publish_observation, daemon=True)
            #     self.publish_observation_thread.start()
            if self.publish_state_thread is None:
                self.publish_state_thread = threading.Thread(target=self.publish_state, daemon=True)
                self.publish_state_thread.start()
            if self.collect_loop is None and self.start_collection:
                self.collect_loop = self.create_timer(1.0/self.FPS_COLLECT, self.collection_loop)
            # self.start_collection = True
        else:
            if self.collect_loop is not None:
                self.collect_loop.cancel()
                self.collect_loop = None
            self.start_inference = False
            self.start_correction = False
            self.stop_infer()
            if self.publish_state_thread is not None:
                self.publish_state_thread.join()
                self.publish_state_thread = None
            if self.publish_observation_thread is not None:
                self.publish_observation_thread.join()
                self.publish_observation_thread = None
            self.go_home()

    def manual_gripper_callback(self,msg):
        if self.execute and self.start_correction:
            self.arm.set_gripper_position(int(msg.data))
            # self.manual_gripper = int(msg.data)
            # print("Gripper callled")

    def manual_servo_callback(self,msg):
        if self.start_correction:
            time_start = time.time()
            if self.execute:
                self.arm.set_servo_cartesian(np.array(msg.data, dtype=np.float32), speed=100, mvacc=1000)
            # self.manual_servo = np.array(msg.data, dtype=np.float32)
            # if 1/(time_end-time_start) > 20:
            self.observation_curr = self.get_observation()
            self.collection_loop()
            time_end = time.time()
            print(1/(time_end-time_start))


    def start_infer(self):
        if self.wrist_camera is None or self.exterior_camera is None:
            print("Waiting for camera images...")
            while self.wrist_camera is None or self.exterior_camera is None:
                time.sleep(0.5)
                print("waiting for camera images...")
        # if self.servo_state is None or self.gripper_state is None:
        #     print("Waiting for robot state...")
        #     while self.servo_state is None or self.gripper_state is None:
        #         time.sleep(0.5)
        #         print("waiting for robot state...")

        if self.infer_thread is None:
            print("Starting inference")
            self.t = 0
            self.frames_recorded = 0
            self.observation_curr = self.get_observation()
            self.action_curr = np.array(self.policy.infer(self.observation_curr)["actions"], dtype=np.float32)

            self.infer_thread = threading.Thread(target=self.inference_loop, daemon=True)
            self.exec_thread = threading.Thread(target=self.execution_loop, daemon=True)
            if self.collect_loop is None:
                self.collect_loop = self.create_timer(1.0/self.FPS_COLLECT, self.collection_loop)

            self.infer_thread.start()
            self.exec_thread.start()

            
    
    def stop_infer(self):
        print("Stopping inference")
        if self.infer_thread is not None:
            self.infer_thread.join()
            self.infer_thread = None
        if self.exec_thread is not None:
            self.exec_thread.join()
            self.infer_thread = None
        if self.collect_loop is not None:
            self.collect_loop.cancel()
            self.collect_loop = None

    def execution_callback(self,msg):
        if msg.data:
            self.get_logger().info('Robot will execute actions.')
            self.execute = True
        else:
            self.get_logger().info('Debug mode only.')
            self.execute = False

    def mode_callback(self,msg):
        self.manual_mode = msg.data
        if self.manual_mode:
            self.start_inference = False
            self.stop_infer()
            if self.publish_state_thread is None and self.start:
                self.publish_state_thread = threading.Thread(target=self.publish_state, daemon=True)
                self.publish_state_thread.start()
            if self.publish_observation_thread is None and self.start:
                self.publish_observation_thread = threading.Thread(target=self.publish_state, daemon=True)
                self.publish_observation_thread.start()
            self.start_correction = True
        if not self.manual_mode:
            self.start_correction = False
            if self.publish_state_thread is not None:
                self.publish_state_thread.join()
                self.publish_state_thread = None
            if self.publish_observation_thread is not None:
                self.publish_observation_thread.join()
                self.publish_observation_thread = None
            if self.start:
                self.start_inference = True
                self.start_infer()

    def wrist_camera_callback(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.wrist_camera = img

    def exterior_camera_callback(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.exterior_camera = img

    def save_callback(self, msg):
        save_data = msg.data
        print(f"Episode length: {self.frames_recorded} frames ********************************")
        if self.start_collection:
            if save_data and self.prev_data is not None:
                self.dataset.save_episode()
                print(f"Episode saved with {self.frames_recorded} frames.")
                self.frames_recorded = 0
                self.prev_data = None
            if not save_data and self.prev_data is not None:
                # HARD RESET: clears the in-memory episode buffer
                self.dataset = LeRobotDataset(root=self.dataset_path,repo_id=self.REPO_NAME,)
                self.get_logger().info("Episode discarded, buffer cleared.")
                self.prev_data = None


    # def discard_callback(self, msg):
    #     discard_data = msg.data
    #     if discard_data and self.prev_data is not None:
    #         # HARD RESET: clears the in-memory episode buffer
    #         self.dataset = LeRobotDataset(root=self.dataset_path,repo_id=self.REPO_NAME,)
    #         self.get_logger().info("Episode discarded, buffer cleared.")
    #         self.prev_data = None

    def failure_success_callback(self, msg):
        self.failure = msg.data

    # def servo_callback(self, msg):
    #     with self.condition_variable:
    #         self.servo_state = np.array(msg.data, dtype=np.float32)

    # def gripper_callback(self, msg):
    #     with self.condition_variable:
    #         self.gripper_state = int(msg.data)
    # =========================
    # Observation
    # =========================
    def get_observation(self,collect=False):
        # if self.collect_thread is not None:
        #     self.collect_thread.join()
        # frames_wrist = self.pipelines[1].wait_for_frames()
        # frames_exterior = self.pipelines[0].wait_for_frames()

        # wrist = frames_wrist.get_color_frame()
        # exterior = frames_exterior.get_color_frame()

        # a = np.asanyarray(wrist.get_data())
        # b = np.asanyarray(exterior.get_data())

        a = self.wrist_camera
        b = self.exterior_camera

        servo_pose = self.arm.get_position()[1]
        # with self.condition_variable:
        #     pose = self.servo_state.copy() # x,y,z,roll,pitch,yaw
        #     g_p = self.gripper_state

        # Convert [-180,180] to [0,360] degrees
        servo_pose[3] = servo_pose[3] % 360
        servo_pose[5] = servo_pose[5] % 360

        # Convert roll, pitch, yaw from degrees to radians
        angles_rad = (np.array(servo_pose[3:6]) * np.pi / 180).tolist()
        servo_state = np.array(servo_pose[:3] + angles_rad, dtype=np.float32)

        # # Combine back into state
        # state = np.concatenate((pose[0:3],angles_rad))
        # print("Observation state:", servo_state)

        _, g_p = self.arm.get_gripper_position()
        g_p = np.array((g_p - 850) / -860)

        observation = {
            "observation/exterior_image_1_left": b,
            "observation/wrist_image_left": a,
            "observation/gripper_position": g_p,
            "observation/joint_position": servo_state,
            "prompt": self.TASK_DESCRIPTION,
        }

        if self.start_correction:
            if self.manual_servo is not None:
                self.arm.set_servo_cartesian(self.manual_servo, speed=100, mvacc=1000)
            if self.manual_gripper is not None:
                self.arm.set_gripper_position(self.manual_gripper)
            


        # if collect:
        #     total_state = np.concatenate((servo_state,np.array([g_p],dtype=np.float32)))
        #     self.collection(total_state)

        return observation
    
    def collection_loop(self):
        # if self.start_collection:

            print("collllllllllllllllllllllllllllllected yayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
            # if self.start_correction:
            #     self.observation_curr = self.get_observation()
            
            obs = self.observation_curr.copy()
            tripod_camera = obs["observation/exterior_image_1_left"]
            wrist_camera = obs["observation/wrist_image_left"]
            gripper_pos = obs["observation/gripper_position"]
            servo_state = obs["observation/joint_position"]

            total_state = np.concatenate((servo_state,np.array([gripper_pos],dtype=np.float32)))

            if self.failure:
                base2 = np.ones_like(tripod_camera) * 255
            else:
                base2 = np.zeros_like(tripod_camera)

            if self.prev_data is not None:
                self.dataset.add_frame(
                    {
                        "joint_position": self.prev_data["joints"],
                        "gripper_position": self.prev_data["gripper"],
                        "actions": total_state,  # This is the "future" state reached
                        "exterior_image_1_left": self.prev_data["base"],
                        "exterior_image_2_left": self.prev_data["base2"],
                        "wrist_image_left": self.prev_data["wrist"],
                        "task": self.TASK_DESCRIPTION,
                    }
                )
                self.frames_recorded += 1

            self.prev_data = {
                "joints": total_state[:6],
                "gripper": total_state[-1:],
                "wrist": wrist_camera,
                "base": tripod_camera,
                "base2": base2,
            }

    # =========================
    # Command Interpolation
    # =========================
    def interpolate_action(self, state, goal):
        delta_increment = (goal - state) / (self.DT * self.CONTROL_HZ * 3)

        for i in range(int(self.DT * self.CONTROL_HZ)):
            start_time = time.perf_counter()
            state += delta_increment
            command = state.copy()
            # print(type(command))
            # print("Command pre:",command)
            command[3] = (command[3]+ 180) % 360 -180
            # print("Command mid:",command)
            command[5] = (command[5]+ 180) % 360 -180

            x, y, z, roll, pitch, yaw = command
            print("Command:",command)
            # print(x, y, z, roll, pitch, yaw)
            if self.execute:
                self.arm.set_servo_cartesian(command, speed=100, mvacc=1000)
            # cmd_auto_servo = Float32MultiArray()
            # cmd_auto_servo.data = command.astype(np.float32).tolist()
            # self.auto_servo_pub.publish(cmd_auto_servo)

            time_left = (1 / self.CONTROL_HZ) - (time.perf_counter() - start_time)
            print("Time left:",time_left)
            time.sleep(max(time_left,0))

    # =========================
    # Action Getter
    # =========================
    def get_action(self, observation_next):
        with self.condition_variable:
            self.t += 1
            self.observation_curr = observation_next
            self.condition_variable.notify()
            action = self.action_curr[self.t - 1, :].copy()
        return action

    # =========================
    # Guided Inference
    # =========================
    def guided_inference(self, policy, observation, action_prev, delay, time_since_last_inference):
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

        v_pi = np.array(policy.infer(observation)["actions"])
        v_pi = v_pi[:H, :self.ROBOT_DOF]  # ensure correct shape

        A = action_prev.copy()
        action_estimate = A*W[:,None] + v_pi*(1-W[:, None])

        return action_estimate[:H, :self.ROBOT_DOF]


    # =========================
    # Inference Loop
    # =========================
    def inference_loop(self):
        Q = deque([self.DELAY_INIT], maxlen=self.BUFFER_SIZE)
        last_collect_time = time.time()

        while self.start_inference:
            
            with self.condition_variable:
                while self.t < self.MIN_EXECUTION_HORIZON:
                    self.condition_variable.wait(timeout=0.5)
                    if not self.start:
                        break
                    if self.manual_mode:
                        break

                time_since_last_inference = self.t
                # Remove actions that have already been executed
                action_prev = self.action_curr[
                    time_since_last_inference:self.PREDICTION_HORIZON
                ].copy() 

                delay = max(Q)
                print("Delay: ", delay)
                obs = self.observation_curr.copy()

            # ---- lock released ----

            action_new = self.guided_inference(
                self.policy,
                obs,
                action_prev,
                delay,
                time_since_last_inference
            )

            self.action_curr[:action_new.shape[0], :] = action_new
            self.t = self.t - time_since_last_inference
            Q.append(self.t)
        print("Inference loop terminated.")

    # =========================
    # Execution Loop
    # =========================
    def execution_loop(self):

        while self.start_inference:
            print("t:", self.t)
            t0 = time.perf_counter()

            observation = self.get_observation()
            command = self.get_action(observation)

            cmd_joint_pose = command[:6].copy()
            cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

            pose = self.arm.get_position()[1]
            # with self.condition_variable:
            #     pose = self.servo_state.copy()

            # Convert [-180,180] to [0,360]
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            state = np.array(pose, dtype=np.float32)

            print("Inference state:", state)
            print("Command pose:", cmd_joint_pose)
            
            # print("Current pose:")
            # print(pose)
            # print("Command pose:")
            # print(cmd_joint_pose)

            # execute smooth motion to target via interpolation
            self.interpolate_action(state, cmd_joint_pose)
            cmd_gripper_pose = (command[6]) * -860 + 850 # unnormalize the gripper action
            if self.execute:
                self.arm.set_gripper_position(cmd_gripper_pose)
            # cmd_auto_gripper = Int32()
            # cmd_auto_gripper.data = int(cmd_gripper_pose)
            # self.auto_gripper_pub.publish(cmd_auto_gripper)

            time_left = self.DT - (time.perf_counter() - t0)
            time.sleep(max(time_left, 0))
        print("Execution loop terminated.")

    def publish_state(self):
        while self.start and self.manual_mode:
            # print(self.arm.get_position()[1])
                servo_state_msg = Float32MultiArray(data=self.arm.get_position()[1])
                gripper_state_msg = Int32(data=0)
                self.gripper_state_pub.publish(gripper_state_msg)
                self.servo_state_pub.publish(servo_state_msg)

                # self.collection_loop()
        print("Publishing loop terminated")

    # def publish_observation(self):
    #     # time_last_obs = time.time()
    #     while self.start and self.manual_mode and self.start_collection:
    #         self.observation_curr = self.get_observation()
    #         # time_since = time.time()-time_last_obs
    #         # time_left = 1/(self.FPS_COLLECT*2) - time_since
    #         # print("Time left: ", time_left)
    #         # time.sleep(0.5)
    #         # time_last_obs = time.time()

    def go_home(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_gripper_mode(0)
        self.arm.set_state(0)
        cmd_joint_pose = [0.0, -90.4, -24.0, 0.0, 61.3, 180.0] 
        cmd_gripper_pose = 850.0
        self.arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=False, wait=True) 
        self.arm.set_gripper_position(cmd_gripper_pose, wait=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()