import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray
from xarm.wrapper import XArmAPI
import time

class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')

        # Start up XArm
        self.arm = XArmAPI("192.168.1.222")
        self.arm.connect()
        if self.arm.get_state() != 0:
            self.arm.clean_error()
            time.sleep(0.5)
        self.arm.motion_enable(enable=True)
        self.arm.set_gripper_enable(enable=True) 
        self.arm.set_gripper_mode(0)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(2.0)
        
        # Subscriber: gui.py start button
        self.start_sub = self.create_subscription(Bool,'/start_button',self.start_callback,10)

        # Subscriber: correction.py, mode toggle (left and right button), True for manual, False for auto
        self.mode_sub = self.create_subscription(Bool,'/manual_mode',self.mode_callback,10)

        # Subscriber: inference.py, "auto" commands
        self.auto_gripper_sub = self.create_subscription(Float32,'/cmd_auto_gripper',self.auto_gripper_callback,10)
        self.auto_servo_sub = self.create_subscription(Float32MultiArray,'/cmd_auto_servo',self.auto_servo_callback,10)

        # Subscriber: correction.py, "manual" commands
        self.manual_gripper_sub = self.create_subscription(Float32,'/cmd_manual_gripper',self.manual_gripper_callback,10)
        self.manual_servo_sub = self.create_subscription(Float32MultiArray,'/cmd_manual_servo',self.manual_servo_callback,10)

        # Publisher: Current robot state
        self.servo_state_pub = self.create_publisher(Float32MultiArray, '/servo_state', 10)
        self.gripper_state_pub = self.create_publisher(Float32, '/gripper_state', 10)

        # Initializations
        self.start = False
        self.current_auto_gripper_cmd = None
        self.current_auto_servo_cmd = None
        self.current_manual_gripper_cmd = None
        self.current_manual_servo_cmd = None
        self.mode = 'auto'  # start in auto mode

        # Timer loop for sending commands to the robot
        self.timer = self.create_timer(0.01, self.control_loop)  # 100Hz
        self.timer_state = self.create_timer(0.1, self.publish_state)  # 10Hz

    def start_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received start signal, beginning control loop.')
        else:
            self.get_logger().info('Received stop signal, halting control loop.')
        self.start = msg.data
   

    def manual_gripper_callback(self, msg):
        self.current_manual_gripper_cmd = msg

    def manual_servo_callback(self, msg):
        self.current_manual_servo_cmd = msg

    def auto_gripper_callback(self, msg):
        self.current_auto_gripper_cmd = msg

    def auto_servo_callback(self, msg):
        self.current_auto_servo_cmd = msg
        

    def mode_callback(self, msg):
        self.mode = 'manual' if msg.data else 'auto'

    def publish_state(self):
        servo_state_msg = Float32MultiArray(data=self.arm.get_position()[1])
        gripper_state_msg = Float32(data=self.arm.get_gripper_position()[1])
        self.servo_state_pub.publish(servo_state_msg)
        self.gripper_state_pub.publish(gripper_state_msg)

    def control_loop(self):

        gripper_cmd = None
        servo_cmd = None
        if self.mode == 'manual' and self.current_manual_gripper_cmd is not None and self.current_manual_servo_cmd is not None:
            gripper_cmd = self.current_manual_gripper_cmd
            servo_cmd = self.current_manual_servo_cmd

        elif self.mode == 'auto' and self.current_auto_gripper_cmd is not None and self.current_auto_servo_cmd is not None:
            gripper_cmd = self.current_auto_gripper_cmd
            servo_cmd = self.current_auto_servo_cmd
        
        if gripper_cmd and servo_cmd and self.start:
            # self.arm.set_servo_cartesian(servo_cmd.data, speed=300, mvacc=2000)
            # self.arm.set_gripper_position(gripper_cmd.data)
            self.get_logger().info(f'Mode: {self.mode}')
            self.get_logger().info(f'Sending gripper command: {gripper_cmd.data}')
            self.get_logger().info(f'Sending servo command: {servo_cmd.data}')

def main(args=None):
    rclpy.init(args=args)
    node = MasterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()