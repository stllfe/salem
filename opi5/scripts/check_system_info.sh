#!/bin/bash

echo "### System info:"
uname -a
echo "CPUs:"
lscpu | grep "Model name" | sed 's/Model name: */ - /'
echo "Number of cores: $(nproc)"
sudo cat /sys/kernel/debug/rknpu/version
