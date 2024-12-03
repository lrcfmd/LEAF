from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filename):
    with Path(filename).open() as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="LEAFeatures",
    version="0.1.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,  # Ensures non-code files are included
    package_data={
        "": ["data/*.json"],  # Specify patterns to include data files
    },
    install_requires=parse_requirements("requirements.txt"),
    description="Local Enviroment-induced Atomic Features (LEAF)",
    author="Andrij Vasylenko",
    url="https://github.com/lrcfmd/LEAF",
    license="MIT",
)

