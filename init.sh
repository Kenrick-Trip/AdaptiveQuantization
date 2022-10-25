docker volume create quantresults
echo "Volume created!"
echo "Building quantizationrunner"
#docker build Quantization -t quantizationrunner
docker build -t quantizationrunner -f Quantization/Dockerfile .
echo "Finished building quantizationrunner"
echo "Building quantizationtester"
#docker build Experiment -t quantizationtester
docker build -t quantizationtester -f Experiment/Dockerfile .
echo "Finished building quantizationtester"

echo "Running quantization"
docker run -v 'quantresults:/resultsets' quantizationrunner
echo "Finished running quantization"