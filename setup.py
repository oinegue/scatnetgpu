from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='scatnetgpu',
      version='0.1.3',
      description='Scattering Network for Python and CUDA',
      long_description=readme(),
      url='http://github.com/oinegue/scatnetgpu',
      author='Eugenio Nurrito',
      author_email='eugi90@gmail.com',
      license='MIT',
      packages=['scatnetgpu'],
      install_requires=[
          'pycuda>=2016.1.2',
          'scikit-cuda>=0.5.1',
          'numpy',
          'scipy',
          'oct2py',
      ],
      include_package_data=True,
      zip_safe=False)
