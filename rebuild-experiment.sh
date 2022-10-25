echo "Building quantizationtester"

docker build -t quantizationtester -f Experiment/Dockerfile .
echo "Finished building quantizationtester"