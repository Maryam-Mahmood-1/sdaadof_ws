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
            'mpipe = some_examples_py.mpipe:main',
        ],
    },
)
