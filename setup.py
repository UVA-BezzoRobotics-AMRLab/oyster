from setuptools import setup, find_packages

setup(
    name="oyster",
    version="0.1.0",
    packages=find_packages(),
<<<<<<< Updated upstream
    install_requires=["numpy", "torch", "matplotlib", "gymnasium", "joblib"],
=======
    install_requires=["numpy", "torch", "matplotlib", "click", "gtimer", "joblib", "gym"],
>>>>>>> Stashed changes
)
