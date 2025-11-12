FROM python:3.12-slim
ARG USER=app UID=1000 GID=1000
RUN groupadd -g $GID $USER && useradd -m -u $UID -g $GID -s /bin/bash $USER
USER $USER
WORKDIR /workspace
CMD ["/bin/bash"]
