experiment_folder=${PWD}"/test_cmaes"
engine_outside_folder=${PWD}"/engine"
engine_inside_folder="/root/.local/nrp/lib/python3.8/site-packages/nrp_core/engines/opensim"
test_outside_folder="/home/alia/Desktop/cespar_initMilestone"
test_inside_folder="/root/.local/nrp/test"

docker pull hbpneurorobotics/nrp-core:opensim_tvb

xhost +si:localuser:root

docker run -ti \
        --network=host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ${experiment_folder}:"/root/nrp-core/examples/neurorobin" \
        -v ${engine_outside_folder}:${engine_inside_folder} \
	-v ${test_outside_folder}:${test_inside_folder} \
        --entrypoint "/bin/bash" -w "/root/nrp-core/examples/neurorobin"\
        hbpneurorobotics/nrp-core:opensim_tvb




