import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    'numpy',
    'ebmlite>=3.0.0',
    'idelib>=3.1.0',
    'pandas>=1.2.4',
    ]

TEST_REQUIRES = [
    'pytest>=4.6',
    # 'mock',
    # 'pytest-cov',
    ]

EXAMPLE_REQUIRES = [
    # 'matplotlib'
    ]

setuptools.setup(
        name='endaq',
        version='1.0.0a1',
        author='Mide Technology',
        author_email='help@mide.com',
        description='A comprehensive, user-centric Python API for working with enDAQ data and devices',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/MideTechnology/endaq-python',
        license='MIT',
        classifiers=['Development Status :: 2 - Pre-Alpha',
                     'License :: OSI Approved :: MIT License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Topic :: Scientific/Engineering',
                     ],
        keywords='ebml binary ide mide endaq',
        # packages=setuptools.find_packages(exclude=('testing',)),
        package_dir={'endaq': './endaq'},
        # package_data={
        #     'idelib': ['schemata/*'],
        # },
        # test_suite='./testing',
        install_requires=INSTALL_REQUIRES,
        extras_require={
            'test': INSTALL_REQUIRES + TEST_REQUIRES,
            'example': INSTALL_REQUIRES + EXAMPLE_REQUIRES,
            },
)
