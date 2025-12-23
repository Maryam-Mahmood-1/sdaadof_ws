from setuptools import setup

package_name = 'pinocchio_test_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={'': ['package.xml']},  # <-- ensures package.xml is installed
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maryam Mahmood',
    maintainer_email='mmahmood.msee23seecs@seecs.edu.pk',
    description='ROS2 Python package to test the Pinocchio library',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pinocchio_test_node = pinocchio_test_pkg.pinocchio_test_node:main',
        ],
    },
)
