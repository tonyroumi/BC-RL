#!/usr/bin/env bash

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "Detected operating system: $OS"

# Download and install CARLA
mkdir carla
cd carla

if [[ "$OS" == "Linux" ]]; then
    echo "Downloading CARLA for Linux..."
    wget -O carla.tar.gz https://tiny.carla.org/carla-0-9-10-1-linux
    wget -O additional_maps.tar.gz https://tiny.carla.org/additional-maps-0-9-10-1-linux
    tar -xf carla.tar.gz
    tar -xf AdditionalMaps_0.9.10.1.tar.gz
    rm carla.tar.gz
    rm AdditionalMaps_0.9.10.1.tar.gz
elif [[ "$OS" == "Windows" ]]; then
    echo "Downloading CARLA for Windows..."
    # Replace with actual Windows download URLs
    wget -O carla.zip https://tiny.carla.org/carla-0-9-10-1-windows
    wget -O AdditionalMaps_0.9.10.1.zip https://tiny.carla.org/additional-maps-0-9-10-1-windows
    # Use appropriate unzip command for Windows
    unzip carla.zip
    unzip AdditionalMaps_0.9.10.1.zip
    rm carla.zip
    rm AdditionalMaps_0.9.10.1.zip
fi

cd ..