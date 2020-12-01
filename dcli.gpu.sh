#docker run -it --rm -p 5000:5000 --entrypoint /bin/bash -v $PWD:/train definition_extraction_app
#docker run -it --rm -p 5000:5000 -v $PWD:/train definition_extraction_app
#docker run -it --rm --name definitionextract -p 5002:5002 docker.crosslang.com/ctlg-manager/definitionextract
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -d -it --rm --name definitionextract -p 5002:5000 definition_extraction_app_gpu
