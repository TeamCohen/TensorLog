from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='tensorlog',
      version='1.2.5',
      description='Differentiable deductive database platform',
      url='https://github.com/TeamCohen/TensorLog',
      author='William Cohen',
      author_email='wcohen@cs.cmu.edu',
      license='Apache 2.0',
      install_requires=[
          # for theano cross-compiler only
          'theano',
          # for debug only
          'ttk', 'Tkinter', 'tkfont',
      ],
      packages=['tensorlog'],
      zip_safe=False)
