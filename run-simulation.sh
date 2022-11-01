docker build -t simulationhost -f Experiment/DockerfileSimulation .
# TODO Build simulation runner
 docker network create --driver bridge inf-net
# Run the simulationhost under the ipalias, "simhost", so any requests can be made to "simhost/"
 docker run -dit --name simhost --network inf-net simulationhost
# TODO Run simulation runner