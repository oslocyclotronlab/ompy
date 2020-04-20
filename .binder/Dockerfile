# The Dockerfile is an attempt to combine
# the Dockerfiles for CodeOcean and MyBinder
# currently one will need to comment/uncomment by hand
# Note: The Dockerfile in the main dir is just a link

# CodeOcean
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cm-super=0.3.4-11 \
        cmake=3.10.2-1ubuntu2.18.04.1 \
        gfortran=4:7.4.0-1ubuntu2.3 \
        libblas-dev=3.7.1-4ubuntu1 \
        liblapack-dev=3.7.1-4ubuntu1 \
        libomp-dev=5.0.1-1 \
        libopenmpi-dev=2.1.1-8 \
        wget=1.19.4-1ubuntu2.2 \
        && rm -rf /var/lib/apt/lists/*


# User spevification needed for MyBinder only
# Configuration required for using Binder
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
ENV NB_USER $NB_USER
ENV HOME /home/${NB_USER}

# create the notebook user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    --gid ${NB_GID} \
    ${NB_USER}

RUN pip install --no-cache -U\
    cython==0.29.14 \
    ipywidgets==7.5.0 \
    matplotlib==3.1.1 \
    notebook==6.0.0 \
    numpy==1.18.1 \
    pandas==0.25.0 \
    pathos==0.2.5 \
    pillow==6.1.0 \
    pymultinest==2.9 \
    scipy==1.4.1 \
    termtables==0.1.0 \
    tqdm==4.35.0 \
    uncertainties==3.1.2

# For CodeOCEAN
# MultiNest is installed in the postInstall script; ENV needs to be set here
#ENV LD_LIBRARY_PATH=$PWD/MultiNest-3.10/lib/:$LD_LIBRARY_PATH
#COPY postInstall /
#RUN /postInstall

# Rest: For MyBinder
WORKDIR ${HOME}
USER root
RUN chown -R 1000:100 ${HOME}

ENV LD_LIBRARY_PATH=/home/jovyan/MultiNest-3.10/lib/:$LD_LIBRARY_PATH
# Due to some cache issue with MyBinder we ought to use COPY instead
# of git clone.
COPY --chown=1000:100 . ompy

RUN cd ompy &&\
    pip install --no-cache -e . && \
    # git submodule update --init --recursive &&\ # now in hooks/post_checkout
    cd ../

USER ${NB_USER}

RUN [ "/bin/bash", "-c", \
      "cd ${HOME} && \
      wget --content-disposition https://github.com/JohannesBuchner/MultiNest/archive/v3.10.tar.gz && \
      tar -xzvf MultiNest-3.10.tar.gz && \
      rm MultiNest-3.10.tar.gz &&\
      cd MultiNest-3.10/build/ && \
      cmake .. && \
      make && \
      cd ../../" ]

# specify jupyterhub configuration
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

EXPOSE 8888

