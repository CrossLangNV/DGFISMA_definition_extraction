BUILD
`
docker build -t definition_extraction_app/retraining/notebook .
`

Run (from root?)
`
docker run -it --rm --gpus '"device=1,"' -p 18888:8888 -v /:/notebook definition_extraction_app/retraining/notebook
`
 
Port forwarding
`
ssh -L 18888:127.0.0.1:18888 <username>@192.168.105.41
`
                