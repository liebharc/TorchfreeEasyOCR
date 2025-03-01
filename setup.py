from io import open
from setuptools import setup

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

setup(
    name='torchfree_ocr',
    packages=['torchfree_ocr'],
    include_package_data=True,
    version='1.0.1',
    install_requires=requirements,
    entry_points={"console_scripts": ["torchfree_ocr= torchfree_ocr.cli:main"]},
    license='Apache License 2.0',
    description='End-to-End English Optical Character Recognition',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/SeldonHZ/torchfree-ocr-en',
    download_url='https://github.com/SeldonHZ/torchfree-ocr-en.git',
    keywords=['ocr optical character recognition deep learning neural network'],
)