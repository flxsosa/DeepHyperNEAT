from setuptools import setup

setup(
    name='deep_hyperneat',
    version='0.1.0',
    author='Sosa, Felix',
    author_email='felixanthonysosa@gmail.com',
    maintainer='Sosa, Felix',
    maintainer_email='felixanthonysosa@gmail.com',
    url='https://github.com/flxsosa/DeepHyperNEAT',
    license="MIT",
    description='A public python implementation of the DeepHyperNEAT system for evolving neural networks.',
    long_description='A public python implementation of the DeepHyperNEAT system for evolving neural networks. Developed by Felix Sosa and Kenneth Stanley. See paper here: https://eplex.cs.ucf.edu/papers/sosa_ugrad_report18.pdf',
    packages=['deep_hyperneat'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.x',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ],
    install_requires=['numpy', 'seaborn', 'graphviz', 'matplotlib']
)
