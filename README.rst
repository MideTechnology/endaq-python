###
WIP
###

This is a work in progress for a python template repo.

This is the target structure.

.. code-block::

    project_name\
        docs\
            source\
                # Sphinx source goes here
        project_name\
            # Code goes here
        tests\
            # Tests go here, can be split further
        .gitignore  # Basic configuration for python, pycharm, mypy, pylint, and any other tools we're using.
        .travis.yml  # runs pytest, maybe coverage, pylint, and mypy.  Python 3.5-dev (3.10 currently) on windows and linux
        pyproject.toml  # config settings for pytest, pylint, mypy, etc.
        README.rst  # Specifically using reStructuredText to better support sphinx
        REQUIREMENTS.txt  # Requirements for development, so lots of packages that may not actually be included code
        setup.py  # Requirements for basic installation and testing as separate requirements
        # other build scripts, or release scripts, etc.


There are several big todo items

- Create some generic files
  - .travis.yml
  - pyproject.toml
  - README.rst  # with instructions on how to setup the repo
  - REQUIREMENTS.txt  # empty, for project run requirements
  - REQUIREMENTS_TESTING.txt  # for testing, to be used by travis when running CI
  - REQUIREMENTS_DEVELOPMENT.TXT  # for development, which will include docs.
  - possibly more requirements files based on i.e. deployment, docs, etc.
  - setup.py  # currently there's discussion about whether or not we want this
- setup some other automated tools, depending on if this is closed or open source.
  - readthedocs supports private documentation repose, which is great.
  - codecoverage would be useful
  - something to do separate inspections for code quality
- make this a cookiecutter template
  - folder structure, of course
  - some things inside files, if possible.