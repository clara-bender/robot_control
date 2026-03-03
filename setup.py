from setuptools import find_packages, setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
    "setuptools<81.0.0,>=71.0.0",        # always include
    # "rclpy",             # ROS 2 Python client library
    # "std_msgs",          # ROS 2 standard messages
    # "geometry_msgs",     # ROS 2 geometry messages
    # "sensor_msgs",       # ROS 2 sensor messages
    # "cv_bridge",         # ROS 2 OpenCV bridge
    "rerun-sdk>=0.24.0,<0.27.0",  # for visualization
    "xarm-python-sdk",   # XArmAPI wrapper
    "numpy",             # numerical operations
    "scipy",             # for Rotation and other numerical functions
    "Pillow",            # for PILImage/ImageTk
    "tk",                # Tkinter (though usually comes with Python)
    # "realsense" might need pyrealsense2
    "pyrealsense2",      # Intel Realsense Python wrapper
    ],
    python_requires='>=3.10',
    zip_safe=True,
    maintainer='admin',
    maintainer_email='csbender@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'master_controller = robot_control.master_controller:main',
            'inference = robot_control.inference:main',
            'correction = robot_control.correction:main',
            'collection = robot_control.collection:main',
            'gui = robot_control.gui:main',
            'cameras = robot_control.cameras:main',
        ],
    },
)
