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

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # =========================
        # User inputs
        # =========================
        self.FPS = 20.0 # still finding upper limit on this
        self.DT = 1.0 / self.FPS
        self.CONTROL_HZ = 140.0 # multiple of 10
        self.PREDICTION_HORIZON = 20
        self.MIN_EXECUTION_HORIZON = 10
        self.ROBOT_DOF = 7

        mutex = threading.Lock()
        self.condition_variable = threading.Condition(mutex)

        self.delay_init = 5
        self.buffer_size = 5

        # =========================
        # Shared State
        # =========================
        self.t = 0
        self.start = False
        self.start_inference = False
        self.start_correction = False
        self.manual_mode = False
        self.infer_thread = None
        self.exec_thread = None
        self.publish_state_thread = None
        self.execute = False


        # =========================
        # Policy Setup
        # =========================
        config = _config.get_config("pi05_xarm_finetune")
        checkpoint_dir = download.maybe_download(
            "/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
        )

        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)


        # =========================
        # XArm Setup
        # =========================
        self.arm = XArmAPI('192.168.1.222')

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

        # self.timer_state = self.create_timer(1/40, self.publish_state)  # 100Hz
        

        print("Ready to go!")

    # =========================
    # Start/Stop
    # =========================
    def start_callback(self, msg):
        self.start = msg.data
        # if self.start:
            
        if self.start and not self.manual_mode:
            self.start_inference = True
            self.start_infer()
        elif self.start and self.manual_mode:
            self.start_correction = True
        else:
            self.start_inference = False
            self.start_correction = False
            self.stop_infer()
            if self.publish_state_thread is not None:
                self.publish_state_thread.join()
            self.go_home()

    def manual_gripper_callback(self,msg):
        if self.execute and self.start_correction:
            self.arm.set_gripper_position(int(msg.data))

    def manual_servo_callback(self,msg):
        if self.execute and self.start_correction:
            self.arm.set_servo_cartesian(np.array(msg.data, dtype=np.float32), speed=100, mvacc=1000)


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
            self.observation_curr = self.get_observation()
            self.action_curr = np.array(self.policy.infer(self.observation_curr)["actions"], dtype=np.float32)

            self.infer_thread = threading.Thread(target=self.inference_loop, daemon=True)
            self.exec_thread = threading.Thread(target=self.execution_loop, daemon=True)

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
            self.publish_state_thread = threading.Thread(target=self.publish_state, daemon=True)
            self.publish_state_thread.start()
            self.start_correction = True
        if not self.manual_mode:
            self.start_correction = False
            self.publish_state_thread.join()
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

    def servo_callback(self, msg):
        with self.condition_variable:
            self.servo_state = np.array(msg.data, dtype=np.float32)

    def gripper_callback(self, msg):
        with self.condition_variable:
            self.gripper_state = int(msg.data)
    # =========================
    # Observation
    # =========================
    def get_observation(self):
        # frames_wrist = self.pipelines[1].wait_for_frames()
        # frames_exterior = self.pipelines[0].wait_for_frames()

        # wrist = frames_wrist.get_color_frame()
        # exterior = frames_exterior.get_color_frame()

        # a = np.asanyarray(wrist.get_data())
        # b = np.asanyarray(exterior.get_data())

        a = self.wrist_camera
        b = self.exterior_camera

        pose = self.arm.get_position()[1]
        # with self.condition_variable:
        #     pose = self.servo_state.copy() # x,y,z,roll,pitch,yaw
        #     g_p = self.gripper_state

        # Convert [-180,180] to [0,360] degrees
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360

        angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
        state = np.array(pose[:3] + angles_rad, dtype=np.float32)

        # # Convert roll, pitch, yaw from degrees to radians
        # angles_rad = (np.array(pose[3:6]) * np.pi / 180)

        # # Combine back into state
        # state = np.concatenate((pose[0:3],angles_rad))
        print("Observation state:", state)

        _, g_p = self.arm.get_gripper_position()
        g_p = np.array((g_p - 850) / -860)

        observation = {
            "observation/exterior_image_1_left": b,
            "observation/wrist_image_left": a,
            "observation/gripper_position": g_p,
            "observation/joint_position": state,
            "prompt": "Pick up the bag and place it on the blue x",
        }

        return observation

    # =========================
    # Command Interpolation
    # =========================
    def interpolate_action(self, state, goal):
        delta_increment = (goal - state) / (self.DT * self.CONTROL_HZ * 6)

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
        Q = deque([self.delay_init], maxlen=self.buffer_size)

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

    # def go_home(self):
    #     self.arm.motion_enable(enable=True)
    #     self.arm.set_mode(0)
    #     self.arm.set_gripper_enable(enable=True)
    #     self.arm.set_gripper_mode(0)
    #     self.arm.set_state(0)
    #     cmd_joint_pose = [0.0, -90.4, -24.0, 0.0, 61.3, 180.0] 
    #     cmd_gripper_pose = 850.0
    #     self.arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=False, wait=True) 
    #     self.arm.set_gripper_position(cmd_gripper_pose, wait=True)
    #     self.arm.motion_enable(enable=True)
    #     self.arm.set_mode(1)
    #     self.arm.set_state(0)

    def publish_state(self):
        while self.start and self.manual_mode:
            # print(self.arm.get_position()[1])
                servo_state_msg = Float32MultiArray(data=self.arm.get_position()[1])
                gripper_state_msg = Int32(data=0)
                self.gripper_state_pub.publish(gripper_state_msg)
                self.servo_state_pub.publish(servo_state_msg)
        print("Publishing loop terminated")

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