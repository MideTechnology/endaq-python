import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "backports.cached-property; python_version<'3.8'",
    "ebmlite>=3.2.0",
    "idelib>=3.2.3",
    "jinja2",
    "numpy>=1.19.5",
    "pandas>=1.3",
    "plotly>=5.3.1",
    "python-dotenv>=0.18.0",
    "requests>=2.25.1",
    "scipy>=1.7.1",
    ]

TEST_REQUIRES = [
    "hypothesis",
    "pytest",
    "pytest-cov",
    "pytest-xdist[psutil]",
    "sympy",
    ]

DOCS_REQUIRES = [
    "sphinx",
    "pydata-sphinx-theme",
    ]

EXAMPLE_REQUIRES = [
    ]

setuptools.setup(
        name='endaq',
        version='1.3.0',
        author='Mide Technology',
        author_email='help@mide.com',
        description='A comprehensive, user-centric Python API for working with enDAQ data and devices',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/MideTechnology/endaq-python',
        license='MIT',
        classifiers=['Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Topic :: Scientific/Engineering',
                     ],
        keywords='ebml binary ide mide endaq',
        packages=setuptools.find_packages(exclude=('tests',)),
        package_dir={'endaq': './endaq'},
        # package_data={
        #     'idelib': ['schemata/*'],
        # },
        # test_suite='./testing',
        project_urls={
            "Bug Tracker": "https://github.com/MideTechnology/endaq-python/issues",
            "Documentation": "https://docs.endaq.com/en/latest/",
            "Source Code": "https://github.com/MideTechnology/endaq-python/tree/main",
            },
        install_requires=INSTALL_REQUIRES,
        extras_require={
            'test': INSTALL_REQUIRES + TEST_REQUIRES,
            'docs': INSTALL_REQUIRES + DOCS_REQUIRES,
            'example': INSTALL_REQUIRES + EXAMPLE_REQUIRES,
            },
)
