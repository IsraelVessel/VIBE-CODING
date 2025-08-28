"""
Environment Setup Script for AI Data Analyzer
Automates the setup process for new installations
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("🔧 Creating virtual environment...")
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created: ./venv")
        
        # Get activation script path
        if os.name == 'nt':  # Windows
            activate_script = Path("venv/Scripts/activate.bat")
            pip_path = Path("venv/Scripts/pip.exe")
        else:  # Unix/Linux/macOS
            activate_script = Path("venv/bin/activate")
            pip_path = Path("venv/bin/pip")
        
        print(f"💡 To activate: {activate_script}")
        return str(pip_path)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return None

def install_dependencies(pip_path=None):
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Use virtual environment pip if available, otherwise system pip
    pip_cmd = pip_path if pip_path and Path(pip_path).exists() else "pip"
    
    try:
        # Install requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        
        # Install package in development mode
        subprocess.run([pip_cmd, "install", "-e", "."], check=True)
        print("✅ Package installed in development mode")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "outputs", 
        "logs",
        "examples",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Directories created")

def setup_configuration():
    """Setup configuration files"""
    print("⚙️ Setting up configuration...")
    
    config_path = Path("config/config.yaml")
    
    # Check if config already exists
    if config_path.exists():
        print("✅ Configuration file already exists")
        return True
    
    # Create default config
    default_config = {
        'analysis': {
            'alpha': 0.05,
            'correlation_threshold': 0.7,
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42
        },
        'visualization': {
            'style': 'seaborn-v0_8-darkgrid',
            'palette': 'viridis',
            'figsize': [12, 8],
            'dpi': 300
        },
        'output': {
            'directory': './outputs',
            'report_format': 'html',
            'include_visualizations': True,
            'save_raw_results': True
        },
        'data': {
            'max_file_size': 500,
            'supported_formats': ['csv', 'xlsx', 'xls', 'json', 'parquet'],
            'auto_dtype': True,
            'parse_dates': True
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './logs/analyzer.log'
        }
    }
    
    # Save configuration
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration file created")
    return True

def create_environment_file():
    """Create .env file template"""
    print("🔐 Creating environment file template...")
    
    env_content = """# AI Data Analyzer Environment Variables

# OpenAI API Key (optional - for advanced NLP features)
# OPENAI_API_KEY=your_openai_api_key_here

# Database Connection Strings (examples)
# POSTGRES_URL=postgresql://user:password@localhost:5432/database
# MYSQL_URL=mysql://user:password@localhost:3306/database
# MONGODB_URL=mongodb://localhost:27017/database

# Custom Configuration
# CONFIG_PATH=./config/custom_config.yaml

# Output Settings
# OUTPUT_DIR=./outputs
# REPORT_FORMAT=html

# Logging
# LOG_LEVEL=INFO
# LOG_FILE=./logs/analyzer.log
"""
    
    env_path = Path(".env.example")
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file template created: .env.example")
    print("💡 Copy to .env and customize for your environment")

def run_basic_test():
    """Run a basic test to verify installation"""
    print("🧪 Running basic functionality test...")
    
    try:
        # Try importing main modules
        sys.path.append(str(Path("src").absolute()))
        
        from ai_data_analyzer import AIDataAnalyzer
        from data_ingestion import DataIngestor
        from prompt_parser import PromptParser
        
        print("✅ All modules import successfully")
        
        # Test basic initialization
        analyzer = AIDataAnalyzer()
        print("✅ AI Data Analyzer initializes correctly")
        
        # Test configuration loading
        if Path("config/config.yaml").exists():
            with open("config/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Configuration loads correctly")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def display_getting_started():
    """Display getting started information"""
    
    print("\\n" + "=" * 60)
    print("🎉 AI Data Analyzer Setup Complete!")
    print("=" * 60)
    
    print("\\n🚀 Getting Started:")
    print("\\n1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\\n2. Try the example:")
    print("   python examples/example_usage.py")
    
    print("\\n3. Interactive mode:")
    print("   python src/ai_data_analyzer.py --interactive")
    
    print("\\n4. Command line usage:")
    print("   python src/ai_data_analyzer.py --data data/sample_customer_data.csv --prompt \\"Describe my data\\"")
    
    print("\\n📖 Documentation:")
    print("   • README.md - Complete documentation")
    print("   • config/config.yaml - Configuration options")
    print("   • examples/ - Usage examples")
    
    print("\\n🔧 Customization:")
    print("   • Edit config/config.yaml for analysis settings")
    print("   • Copy .env.example to .env for environment variables")
    print("   • Add your own data to the data/ directory")
    
    print("\\n💡 Example Prompts:")
    prompts = [
        "Describe my data",
        "Show correlations between all variables", 
        "Find customer segments",
        "Predict sales using available features",
        "Detect anomalies in the data"
    ]
    
    for prompt in prompts:
        print(f"   • \\"{prompt}\\"")
    
    print("\\n🆘 Need help?")
    print("   • Check README.md for detailed documentation")
    print("   • Run with --help for command line options")
    print("   • Use 'help' command in interactive mode")

def main():
    """Main setup function"""
    
    print("🤖 AI Data Analyzer - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup configuration
    setup_configuration()
    
    # Create environment file
    create_environment_file()
    
    # Ask about virtual environment
    if not Path("venv").exists():
        response = input("\\n🔧 Create virtual environment? (recommended) [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            pip_path = create_virtual_environment()
            if pip_path:
                install_dependencies(pip_path)
            else:
                print("⚠️ Virtual environment creation failed, using system Python")
                install_dependencies()
        else:
            print("⚠️ Using system Python - consider using virtual environment")
            install_dependencies()
    else:
        print("✅ Virtual environment already exists")
        # Try to use existing venv
        if os.name == 'nt':  # Windows
            pip_path = "venv/Scripts/pip.exe"
        else:  # Unix/Linux/macOS
            pip_path = "venv/bin/pip"
        
        if Path(pip_path).exists():
            install_dependencies(pip_path)
        else:
            install_dependencies()
    
    # Run basic test
    if run_basic_test():
        display_getting_started()
    else:
        print("\\n❌ Setup encountered issues. Please check the error messages above.")
        print("💡 Try running the setup steps manually or check the documentation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n🛑 Setup interrupted by user")
    except Exception as e:
        print(f"\\n❌ Setup failed: {str(e)}")
        print("💡 Please check the error and try again, or set up manually using README.md")
