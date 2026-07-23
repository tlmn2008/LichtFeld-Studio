# CoreX build environment (no secrets). Source before building/testing.
. /etc/profile.d/corex.sh
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/corex/lib64:/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/usr/local/corex/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
