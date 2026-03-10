import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32
from sensor_msgs.msg import Joy
import numpy as np
import tkinter as tk
from scipy.spatial.transform import Rotation

class CorrectionNode(Node):
    def __init__(self):
        super().__init__('correction_node')

        # Joystick subscription
        self.joy_sub = self.create_subscription(Joy, 'spacenav/joy', self.joystick_callback, 10)

        # State subscriptions
        self.servo_sub = self.create_subscription(Float32MultiArray, '/servo_state', self.servo_state_callback, 10)
        self.gripper_sub = self.create_subscription(Int32, '/gripper_state', self.gripper_state_callback, 10)

        # Subscriber: gui.py start button
        # self.start_sub = self.create_subscription(Bool,'/start_button',self.correction_callback,10)

        # Publisher to send commands to the arm
        self.gripper_pub = self.create_publisher(Float32, '/cmd_manual_gripper', 10)
        self.servo_pub = self.create_publisher(Float32MultiArray, '/cmd_manual_servo', 10)
        self.mode_pub = self.create_publisher(Bool, '/manual_mode',10)

        # Initializations
        self.joystick_msg = None
        self.latest_axes = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.servo_state = None
        self.gripper_state = None
        self.gripper_position = 0.0
        self.gripper_target = 0.0
        self.manual_mode = False
        self.auto_closing = False
        self.auto_open = False
        self.dt = 1.0/30.0

        # --- GUI for the gripper slider ---
        self.root = tk.Tk()
        self.root.title("Gripper Control")
        self.root.geometry("1500x300")  # Larger window
        
        self.slider = tk.Scale(
            self.root,
            from_=0, to=1,
            resolution=0.01,
            orient='horizontal',
            label='Gripper Open/Close',
            command=self.update_gripper,
            length=1700,  # Longer slider
            width=100,     # Thicker slider
            font=('Arial', 16, 'bold'),  # Larger font
            troughcolor='#E0E0E0',  # Light gray background
            sliderlength=80  # Larger slider handle
        )
        self.slider.pack(fill=tk.X, expand=True, padx=20, pady=50)  # More padding
        self.slider.set(0)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.gripper_position = 0.0

        self.key_step = 0.04
        self.root.focus_force()
        self.root.bind("<Left>", self.on_left_key)
        self.root.bind("<Right>", self.on_right_key)
        self.root.update()

        # Simulate correction with a timer (for demo)
        self.timer = self.create_timer(1.0/20, self.correction_callback)  # Run every 1 second

    def servo_state_callback(self, msg):
        print("servo callback called")
        self.servo_state = np.array(msg.data, dtype=np.float32)
    
    def gripper_state_callback(self, msg):
        print("gripper callback called")
        self.gripper_state = int(msg.data)

    def joystick_callback(self, msg: Joy):
        """
        Called by timer_callback
        """
        self.latest_axes = np.array(msg.axes[:6])  # [x, y, z, roll, pitch, yaw]
        self.joystick_msg = msg
        left_button = bool(msg.buttons[0])
        right_button = bool(msg.buttons[1])

        if left_button and not self.manual_mode:
            self.manual_mode = True
            self.mode_pub.publish(Bool(data=True))
        if right_button and self.manual_mode:
            self.manual_mode = False
            self.servo_state = None
            self.gripper_state = None
            self.mode_pub.publish(Bool(data=False))

    def correction_callback(self):

        if self.servo_state is None:
            print("Not getting servo state")
            if  self.gripper_state is None:
                print("Not getting gripper state")
            return

        # Update gripper for up/down key presses
        if self.auto_closing:
            current_val = self.slider.get()
            if current_val < self.gripper_target:
                new_val = min(self.gripper_target, current_val + 0.02)
                self.slider.set(new_val)
                self.update_gripper(new_val)
            else:
                self.auto_closing = False

        if self.auto_open:
            current_val = self.slider.get()
            if current_val > self.gripper_target:
                new_val = max(self.gripper_target, current_val - 0.02)
                self.slider.set(new_val)
                self.update_gripper(new_val)
            else:
                self.auto_open = False

        if self.joystick_msg is None:
            print("Waiting for joystick message")
            return
        
        # 1) Get current xArm pose
        curr_pose = self.servo_state.tolist()
        curr_pose = np.array(curr_pose)
        curr_euler = curr_pose[3:] 
        curr_quat = Rotation.from_euler('xyz', curr_euler, degrees=True)

        # 2) Get cartesian input from the SpaceMouse
        scale_linear = 140.0
        scale_angular = 40.0
        vx, vy, vz, wx, wy, wz = self.latest_axes * np.array([scale_linear]*3 + [scale_angular]*3)

        # 3. Calculate the rotation delta from SpaceMouse (in radians)
        # angular_velocity * dt
        delta_euler = np.array([wx, wy, wz]) * self.dt * (np.pi / 180.0)
        delta_quat = Rotation.from_rotvec(delta_euler)
             
        # 4. Apply the delta (Matrix multiplication handles the rotation)
        new_quat = delta_quat * curr_quat
        new_euler = new_quat.as_euler('xyz', degrees=True)

        new_xyz = curr_pose[:3] + np.array([vx, vy, vz]) * self.dt

        # 5. Combine with new XYZ positions
        cmd_manual_servo = np.concatenate([new_xyz, new_euler])

        # 6) Convert the slider [0..1] to a gripper command (0 => 850, 1 => -10)
        cmd_manual_gripper = 850 - 860 * self.gripper_position

        # 7) Publish the gripper and servo commands
        self.gripper_pub.publish(Float32(data=cmd_manual_gripper))
        self.servo_pub.publish(Float32MultiArray(data=cmd_manual_servo))
        self.get_logger().info(f"Published Manual Gripper Command: {cmd_manual_gripper}")
        self.get_logger().info(f"Published Manual Servo Command: {cmd_manual_servo}")

        # Final step: update GUI & remember button states
        self.root.update()

    def on_close(self):
        self.root.quit()

    def update_gripper(self, value):
        self.gripper_position = float(value)

    def on_left_key(self, event=None):
        self.gripper_target = 0.0
        self.auto_open = True

    def on_right_key(self, event=None):
        self.gripper_target = 0.94
        self.auto_closing = True


def main(args=None):
    rclpy.init(args=args)
    node = CorrectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()