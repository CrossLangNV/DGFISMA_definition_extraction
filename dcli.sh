#docker run -it --rm -p 5000:5000 --entrypoint /bin/bash -v $PWD:/train definition_extraction_app
docker run -it --rm -p 5000:5000 -v $PWD:/train definition_extraction_app
