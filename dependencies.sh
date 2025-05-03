# Install dependencies on the HPC image
# 1. Update package manager cache (optional but recommended)
sudo dnf check-update

# 2. Ensure base Development Tools are installed (provides g++, make, etc.)
#    This group should already be installed on the HPC image, but running this ensures it.
sudo dnf groupinstall -y "Development Tools"

# 3. Install Python 3.8 Development Headers (provides Python.h)
sudo dnf install -y python38-devel

# 4. Install Ninja build system
sudo dnf install -y ninja-build

echo "Required build dependencies should now be installed."