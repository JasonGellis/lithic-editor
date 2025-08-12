"""
Setup configuration for the Lithic Editor and Annotator package.

This package wraps the proven working lithic editor implementation
with a clean, installable package structure.
"""

from setuptools import setup, find_packages
import os

def get_version():
    """Extract version from package __init__.py file."""
    version_file = os.path.join(os.path.dirname(__file__), 'lithic_editor', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

def get_long_description():
    """Get long description from README file."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Lithic Editor and Annotator - Archaeological image processing tool"

def get_requirements():
    """Get requirements from requirements.txt file."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    # Package metadata
    name='lithic-editor',
    version=get_version(),
    author='Jason Jacob Gellis',
    author_email='jg760@cam.ac.uk',
    description='Archaeological image processing tool for lithic drawings',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/user/lithic-editor',
    
    # Package discovery
    packages=find_packages(),
    include_package_data=True,
    
    # Include the working source files
    package_data={
        '': ['*.py'],  # Include all Python files
    },
    
    # Dependencies
    install_requires=get_requirements(),
    python_requires='>=3.7',
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'lithic-editor=lithic_editor.cli.main:main',
        ],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords=[
        'archaeology', 'lithic', 'image-processing', 'computer-vision',
        'archaeological-tools', 'stone-tools', 'artifact-analysis'
    ],
    
    # Zip safety
    zip_safe=False,
)