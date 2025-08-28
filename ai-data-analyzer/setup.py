"""
Setup script for AI Data Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-data-analyzer",
    version="1.0.0",
    description="AI-powered data analysis tool with natural language prompts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Data Solutions",
    author_email="info@aidatasolutions.com",
    url="https://github.com/your-org/ai-data-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "streamlit>=1.25.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ai-analyzer=ai_data_analyzer:main",
            "ai-analyzer-interactive=ai_data_analyzer:InteractiveAnalyzer.run",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    keywords="ai, data analysis, machine learning, automation, natural language",
    project_urls={
        "Bug Reports": "https://github.com/your-org/ai-data-analyzer/issues",
        "Source": "https://github.com/your-org/ai-data-analyzer",
        "Documentation": "https://github.com/your-org/ai-data-analyzer#readme",
    },
)
