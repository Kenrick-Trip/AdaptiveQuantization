docker volume create quantresults
echo "Volume created!"
echo "Building quantizationrunner"
docker build Quantization -t quantizationrunner
echo "Finished building quantizationrunner"
echo "Building quantizationtester"
docker build Experiment -t quantizationtester
echo "Finished building quantizationtester"

echo "Running quantization"
docker run -v quantresults:/resultsets quantizationrunner
echo "Finished running quantization"