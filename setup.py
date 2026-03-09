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
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admin',
    maintainer_email='csbender@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
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
