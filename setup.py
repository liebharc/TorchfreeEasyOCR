from setuptools import setup, find_packages

setup(
    name='liteOCR',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python-headless',
        'onnxruntime',
        'Pillow',
        'numpy'
    ],
    include_package_data=True, 
)