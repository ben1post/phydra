from setuptools import setup, find_packages
import versioneer

requirements = [
    "attrs >=18.1.0",
    "dask",
    "numpy",
    "xarray >=0.10.0",
    "zarr >=2.3.0",
    "xarray-simlab",
    "scipy",
    "xso",
]

setup(
    name='phydra',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A library of marine ecosystem model built with XSO, an extension of xarray-simlab",
    license="BSD",
    author="Benjamin Post",
    author_email='ben@anoutpost.com',
    url='https://github.com/ben1post/phydra',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'phydra=phydra.cli:cli'
        ]
    },
    install_requires=requirements,
    tests_require=["pytest >= 3.3.0"],
    keywords='xarray-simlab python xarray modelling simulation framework ode',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
