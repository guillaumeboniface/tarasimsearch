# Tara similarity search
This repository is a proof of concept similarity search engine built for [EcoTaxa](https://ecotaxa.obs-vlfr.fr/), using the ZooScanv2 dataset containing about 1.5M images and [Faiss](https://github.com/facebookresearch/faiss).

## Build the container image
Make sure you have docker installed and from the repository do 
> docker build --tag tarasimsearch:1.0 .

## Run the docker image
> docker run -it -p 8080:8080 tarasimsearch:1.0
You should be able to open localhost:8080 and use the website for a similarity search

