from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'scikit-learn',
    'flask',
    'flask-script',
    'flask-bootstrap',
    'werkzeug',
    'bokeh',
    'Jinja2',
]

setup(
    name='gait-calibrate',
    version='1.1',
    description='A Python toolkit for personalized gait calibration',
    author='Akara Supratak',
    author_email='as12212@imperial.ac.uk',
    packages=find_packages(),
    install_requires=install_requires,
)