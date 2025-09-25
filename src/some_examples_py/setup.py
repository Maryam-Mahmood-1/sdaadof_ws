from setuptools import find_packages, setup

package_name = 'some_examples_py'

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
    maintainer='maryam-mahmood',
    maintainer_email='maryam-mahmood@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = some_examples_py.simple_publisher:main',
            'my_action_server = some_examples_py.my_action_server:main',
            'gui_trajectory = some_examples_py.gui_trajectory:main',
            'gui_trajectory_2 = some_examples_py.gui_trajectory_2:main',
            'sim_driver = some_examples_py.sim_driver:main',
            'gui_trajectory_3 = some_examples_py.gui_trajectory_3:main',
            'mpipe = some_examples_py.mpipe:main',
            'z_torque_aggregator = some_examples_py.z_torque_aggregator:main',
            'rviz_torque_text = some_examples_py.rviz_torque_text:main', 
        ],
    },
)
