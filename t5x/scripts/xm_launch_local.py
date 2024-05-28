import os
import tempfile

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import copybara
LOCAL_REPO_PATH = "/Users/tsimur.hadeliya/code/playground/t5x_generative_retrieval"

def main(args):
    with xm_local.create_experiment(experiment_title="local_t5_run") as experiment:
        staging = os.path.join(LOCAL_REPO_PATH, "experiments")
        tempfile.mkdtemp(dir=staging)

        executor = xm_local.Local()
        docker_instructions = [
            'RUN apt-get install apt-transport-https ca-certificates gnupg',
            (
                'RUN echo "deb'
                ' [signed-by=/usr/share/keyrings/cloud.google.gpg]'
                ' http://packages.cloud.google.com/apt cloud-sdk main" |'
                ' tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&'
                ' curl'
                ' https://packages.cloud.google.com/apt/doc/apt-key.gpg |'
                ' apt-key --keyring /usr/share/keyrings/cloud.google.gpg '
                ' add - && apt-get update -y && apt-get install'
                ' google-cloud-cli -y'
            ),
            "RUN git clone --branch=main https://github.com/google-research/t5x",
            # 'COPY /Users/tsimur.hadeliya/code/playground/t5x_generative_retrieval/ t5x',
            "WORKDIR t5x",
            (
                'RUN python3 -m pip install -e "." -f'
                ' https://storage.googleapis.com/jax-releases/libtpu_releases.html'
            )
        ]
        entrypoint=xm.CommandList([
            'export T5X_DIR=.',
            (
               'python3 ${T5X_DIR}/t5x/main.py '
               f'--run_mode=eval '
               '--gin.MODEL_DIR=${MODEL_DIR} '
               '--tfds_data_dir=${TFDS_DATA_DIR} '
               '--undefok=seqio_additional_cache_dirs '
               '--seqio_additional_cache_dirs=${SEQIO_CACHE_DIRS} '
            )
        ])
        container = xm.python_container(
            executor.Spec(),
            path=".",
            base_image="python:3.10",
            docker_instructions=docker_instructions,
            entrypoint=entrypoint,
        )

        [executable] = experiment.package([container])
        experiment.add(
            xm.Job(executable=executable, executor=executor)
        )

if __name__ == "__main__":
    app.run(lambda args: main(args))