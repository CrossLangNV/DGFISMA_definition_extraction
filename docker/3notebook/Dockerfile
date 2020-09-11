#Make accessible with jupyter notebook
FROM definition_extraction_app/base/gpu

EXPOSE 8888

VOLUME /notebook
WORKDIR /notebook

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip='0.0.0.0'"]
