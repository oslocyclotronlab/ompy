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

RUN pip install -U --no-cache-dir \
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

RUN [ "/bin/bash", "-c", \
      "wget --content-disposition https://github.com/JohannesBuchner/MultiNest/archive/v3.10.tar.gz && \
      tar -xzvf MultiNest-3.10.tar.gz && \
      rm MultiNest-3.10.tar.gz && \
      cd MultiNest-3.10/build/ && \
      cmake .. && \
      make && \
      cd ../../" ]
# MultiNest is installed in the postInstall script
ENV LD_LIBRARY_PATH=/MultiNest-3.10/lib/:$LD_LIBRARY_PATH

# For CodeOCEAN
# COPY postInstall /
# RUN /postInstall

# Rest is for MyBinder

# Due to some cache issue with MyBinder we ought to use COPY instead
# of git clone.
COPY --chown=1000:100 . ompy
# REMBEBER TO checkout the BRANCH you want
RUN cd ompy &&\
    # git submodule update --init --recursive &&\ # now in hooks/post_checkout
    pip install -e .

# create a user, since we don't want to run as root
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"
ENV HOME=/home/$NB_USER
WORKDIR $HOME
USER $NB_UID

COPY --chown=$NB_USER:$NB_GID start-notebook.sh /home/$NB_USER

EXPOSE 8888

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# Copy local files as late as possible to avoid cache busting
COPY start.sh start-notebook.sh start-singleuser.sh /usr/local/bin/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID


