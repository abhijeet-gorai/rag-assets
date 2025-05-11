import os
import io
import sys
from typing import List, Optional, Type
from dotenv import load_dotenv
load_dotenv(override=True)

import docker
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

DOCKER_HOST = os.getenv("DOCKER_HOST", None)

class CodeInterpreterSchema(BaseModel):
    """Input for CodeInterpreterTool."""

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container. ALWAYS PRINT the final result and the output of the code",
    )

    libraries_used: List[str] = Field(
        ...,
        description="List of libraries used in the code with proper installing names separated by commas. Example: numpy,pandas,beautifulsoup4",
    )


class CodeInterpreterTool(BaseTool):
    name: str = "Code Interpreter"
    description: str = "Interprets Python3 code strings with a final print statement."
    args_schema: Type[BaseModel] = CodeInterpreterSchema
    default_image_tag: str = "code-interpreter:latest"
    code: Optional[str] = None
    user_dockerfile_path: Optional[str] = None
    unsafe_mode: bool = False
    container_name: str = "code-interpreter"
    new_container_for_each_run: bool = False

    @staticmethod
    def _get_current_file_path():
        spec = os.path.abspath(__file__)
        return os.path.dirname(spec)

    def _verify_docker_image(self) -> None:
        """
        Verify if the Docker image is available. Optionally use a user-provided Dockerfile.
        """
        client = docker.DockerClient(base_url=DOCKER_HOST)

        try:
            client.images.get(self.default_image_tag)

        except docker.errors.ImageNotFound:
            if self.user_dockerfile_path and os.path.exists(self.user_dockerfile_path):
                dockerfile_path = self.user_dockerfile_path
            else:
                current_file_path = self._get_current_file_path()
                dockerfile_path = os.path.join(
                    current_file_path, "code_interpreter_dockerfile"
                )
                if not os.path.exists(dockerfile_path):
                    raise FileNotFoundError(
                        f"Dockerfile not found in {dockerfile_path}"
                    )

            client.images.build(
                path=dockerfile_path,
                tag=self.default_image_tag,
                rm=True,
                platform="linux/amd64"
            )

    def _run(self, **kwargs) -> str:
        code = kwargs.get("code", self.code)
        libraries_used = kwargs.get("libraries_used", [])

        if self.unsafe_mode:
            return self.run_code_unsafe(code, libraries_used)
        else:
            return self.run_code_in_docker(code, libraries_used)

    def _install_libraries(
        self, container: docker.models.containers.Container=None, libraries: List[str]=[]
    ) -> None:
        """
        Install missing libraries in the Docker container
        """
        if container is None:
            client = docker.DockerClient(base_url=DOCKER_HOST)
            container = client.containers.get(self.container_name)
        for library in libraries:
            container.exec_run(f"pip install {library}")

    def _init_docker_container(self, env_variables=None) -> docker.models.containers.Container:
        client = docker.DockerClient(base_url=DOCKER_HOST)
        current_path = os.getcwd()

        # Check if the container is already running
        try:
            existing_container = client.containers.get(self.container_name)
            existing_container.stop()
            existing_container.remove()
        except docker.errors.NotFound:
            pass  # Container does not exist, no need to remove

        return client.containers.run(
            self.default_image_tag,
            detach=True,
            tty=True,
            working_dir="/workspace",
            name=self.container_name,
            environment=env_variables,
            volumes={current_path: {"bind": "/workspace", "mode": "rw"}},  # type: ignore
        )
    
    def _check_container_running(self):
        client = docker.DockerClient(base_url=DOCKER_HOST)
        try:
            container = client.containers.get(self.container_name)
            return container.status == 'running', container
        except docker.errors.NotFound:
            return False, None

    def run_code_in_docker(self, code: str, libraries_used: List[str], env_variables:dict[str, str] | list[str] | None = None) -> str:
        container_running, container = self._check_container_running()
        if self.new_container_for_each_run or (not container_running):
            self._verify_docker_image()
            container = self._init_docker_container(env_variables=env_variables)
        
        self._install_libraries(container, libraries_used)
        code = code.replace('"', "'")
        cmd_to_run = f'python3 -c "{code}"'
        exec_result = container.exec_run(cmd_to_run)

        if self.new_container_for_each_run:
            container.stop()
            container.remove()

        if exec_result.exit_code != 0:
            print(exec_result.output.decode('utf-8'))
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return exec_result.output.decode("utf-8")

    def run_code_unsafe(self, code: str, libraries_used: List[str]) -> str:
        """
        Run the code directly on the host machine (unsafe mode).
        """
        # Install libraries on the host machine
        for library in libraries_used:
            os.system(f"pip install {library}")

        # Execute the code
        try:
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
            exec_locals = {}
            exec(code, {}, exec_locals)
            sys.stdout = sys.__stdout__
            console_output = output_buffer.getvalue()
            output_buffer.close()
            return console_output
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
    def stop_and_remove_container(self):
        container_running, container = self._check_container_running()
        if container_running:
            container.stop()
            container.remove()
