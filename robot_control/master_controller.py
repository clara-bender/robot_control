import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist  # or use std_msgs for simple control
from std_msgs.msg import Bool

class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')
        self.start = False

        self.start_sub = self.create_subscription(
            Bool,
            '/start',
            self.start_callback,
            10
        )   
        
        # Example subscriber: manual commands
        self.manual_sub = self.create_subscription(
            Twist,
            '/cmd_manual',
            self.manual_callback,
            10
        )

        self.mode_sub = self.create_subscription(
            Bool,
            '/cmd_mode',
            self.mode_callback,
            10
        )

        # Example subscriber: auto commands
        self.auto_sub = self.create_subscription(
            Twist,
            '/cmd_auto',
            self.auto_callback,
            10
        )

        self.current_manual_cmd = None
        self.current_auto_cmd = None
        self.mode = 'auto'  # start in auto mode

        # Timer loop for sending commands to the robot
        self.timer = self.create_timer(0.01, self.control_loop)  # 100Hz

    def start_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received start signal, beginning control loop.')
        else:
            self.get_logger().info('Received stop signal, halting control loop.')
        self.start = msg.data
   

    def manual_callback(self, msg):
        self.current_manual_cmd = msg

    def auto_callback(self, msg):
        self.current_auto_cmd = msg

    def mode_callback(self, msg):
        self.mode = 'manual' if msg.data else 'auto'

    def control_loop(self):
        # Only send commands from active mode
        cmd = None
        if self.mode == 'manual' and self.current_manual_cmd:
            cmd = self.current_manual_cmd
        elif self.mode == 'auto' and self.current_auto_cmd:
            cmd = self.current_auto_cmd
        
        if cmd and self.start:  # Only send if we have a command and start signal
            # TODO: call arm.set_servo_cartesian(...) here
            self.get_logger().info(f'Sending command: {cmd.linear.x}, {cmd.linear.y}, {cmd.linear.z}')

def main(args=None):
    rclpy.init(args=args)
    node = MasterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()