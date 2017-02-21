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
      install_requires=['numpy','scipy'],
      extras_require={
        'xc-theano': ['theano'],
        'xc-tensorflow': ['tensorflow'],
        'debug':['ttk', 'Tkinter', 'tkfont'],
        },
      packages=['tensorlog'],
      zip_safe=False)
