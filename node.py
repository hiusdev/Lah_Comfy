import subprocess, os
import tarfile
import os
import shutil
import folder_paths
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np


def parse_args(config):
    args = []

    for k, v in config.items():
        if k.startswith("_"):
            args.append(f"{v}")
        elif isinstance(v, str) and v is not None:
            args.append(f"--{k}={v}")
        elif isinstance(v, bool) and v:
            args.append(f"--{k}")
        elif isinstance(v, float) and not isinstance(v, bool):
            args.append(f"--{k}={v}")
        elif isinstance(v, int) and not isinstance(v, bool):
            args.append(f"--{k}={v}")

    return args


def delete_folder(folder_path):
    """
    Xóa một thư mục (có hoặc không có nội dung).

    :param folder_path: Đường dẫn tới thư mục cần xóa.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Đã xóa thư mục và toàn bộ nội dung: {folder_path}")
    except FileNotFoundError:
        print(f"Thư mục không tồn tại: {folder_path}")
    except Exception as e:
        print(f"Lỗi khi xóa thư mục: {e}")


def delete_file(file_path):
    """
    Xóa một file.

    :param file_path: Đường dẫn tới file cần xóa.
    """
    try:
        os.remove(file_path)
        print(f"Đã xóa file: {file_path}")
    except FileNotFoundError:
        print(f"File không tồn tại: {file_path}")
    except Exception as e:
        print(f"Lỗi khi xóa file: {e}")


def extract_tar_file(tar_path, extract_to):
    """
    Giải nén file .tar đến một thư mục cụ thể.

    :param tar_path: Đường dẫn tới file .tar cần giải nén.
    :param extract_to: Thư mục đích để giải nén file.
    """
    # Kiểm tra nếu file tồn tại
    if not os.path.exists(tar_path):
        print(f"File không tồn tại: {tar_path}")
        return

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(extract_to, exist_ok=True)

    # Giải nén file .tar
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_to)
            print(f"Giải nén thành công: {tar_path} tới {extract_to}")
    except Exception as e:
        print(f"Lỗi khi giải nén: {e}")


def download_model(url, folder_paths, name):
    try:
        # Prepare the file name and configuration for aria2c
        file_name = name + ".tar"
        aria2_config = {
            "console-log-level": "error",
            "summary-interval": 10,
            "continue": True,
            "max-connection-per-server": 16,
            "min-split-size": "1M",
            "split": 16,
            "dir": folder_paths,
            "out": file_name,
            "_url": url,
        }

        # Parse the arguments for aria2c
        aria2_args = parse_args(aria2_config)

        # Run the aria2c command
        result = subprocess.run(["aria2c", *aria2_args], capture_output=True, text=True)

        file_path = None
        # Check if the process was successful
        if result.returncode != 0:
            print("error")
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        else:
            file_path = os.path.join(folder_paths, file_name)

        # Return the full path to the downloaded file
        return file_path
    except subprocess.CalledProcessError as e:
        # Log detailed error information
        print(f"Aria2c failed with exit code {e.returncode}")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stdout: {e.output}")
        print(f"Stderr: {e.stderr}")
        return None
        # raise  # Re-raise the exception for further handling
    except Exception as e:
        # Handle other exceptions (e.g., invalid arguments, IO errors)
        print(f"An error occurred: {e}")
        raise


def move_and_rename_file(src_path, dest_dir, new_name):

    # Kiểm tra nếu file nguồn tồn tại
    if not os.path.exists(src_path):
        print(f"File không tồn tại: {src_path}")
        return

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_dir, exist_ok=True)

    # Xây dựng đường dẫn đích với tên mới
    dest_path = os.path.join(dest_dir, new_name)

    # Di chuyển và đổi tên file
    try:
        shutil.move(src_path, dest_path)
        print(f"Đã di chuyển file tới: {dest_path}")
    except Exception as e:
        print(f"Lỗi khi di chuyển file: {e}")


def download_lora(url, lora_name):
    try:
        models_dir = folder_paths.models_dir
        lora_dir = os.path.join(models_dir, "loras")
        lora_name_safetensors = lora_name + ".safetensors"

        if os.path.exists(os.path.join(lora_dir, lora_name_safetensors)):
            print("load done")
            return lora_name_safetensors

        download_path = os.path.join(models_dir, "download")
        path_file = download_model(url, download_path, lora_name)
        print(path_file)
        if os.path.exists(path_file):
            extra_to = os.path.join(download_path, lora_name)
            if os.path.exists(path_file):
                extract_tar_file(path_file, extra_to)
                lora_path = os.path.join(
                    extra_to, "output/flux_train_replicate/lora.safetensors"
                )
                print(lora_path)
                move_and_rename_file(lora_path, lora_dir, lora_name_safetensors)
                delete_file(path_file)
                delete_folder(extra_to)
        else:
            raise ValueError("Lora download failed")

        if os.path.exists(os.path.join(lora_dir, lora_name_safetensors)):
            return lora_name_safetensors
        else:
            raise ValueError("Lora was not found")
    except Exception as e:
        # Log the exception details or handle them appropriately
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception after handling it for further debugging or propagation


class LoraDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "clip": (
                    "CLIP",
                    {
                        "default": None,
                        "tooltip": "The CLIP model the LoRA will be applied to.",
                    },
                ),
                "url": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "lora_name": (
                    "STRING",
                    {"default": "hius_123", "multiline": False},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the CLIP model. This value can be negative.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    CATEGORY = "LahTeam/Download"

    def load_lora(self, model, clip, url, lora_name, strength_model, strength_clip):
        try:
            lora_path = download_lora(url, lora_name)
            return ALL_NODE["LoraLoader"]().load_lora(
                model, clip, lora_path, strength_model, strength_clip
            )
        except Exception as e:
            # Log the exception details or handle them appropriately
            print(f"An error occurred: {e}")
            raise  # Re-raise the exception after handling it for further debugging or propagation
            # return


class ImageWebHook:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "url": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "id": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),

            }
        }

    RETURN_TYPES = ()
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "image_hook"
    CATEGORY = "LahTeam/Hook"
    OUTPUT_NODE = True

    def image_hook(self, images, url, id, id_task):
        try:
            for batch_number, image in enumerate(images):
                # Process the image and convert it to the desired format
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Save the image into a BytesIO buffer
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)

                # Encode the image to Base64
                img_data = buffer.getvalue()
                img_str = base64.b64encode(img_data).decode("utf-8")

                # Prepare the payload and headers
                payload = {"image": img_str, "id": id}
                headers = {"Content-Type": "application/json"}

                # Make the POST request
                response = requests.post(url, json=payload, headers=headers)

                # Check the response for errors
                if response.status_code != 200:
                    raise requests.exceptions.RequestException(
                        f"Error {response.status_code}: {response.text}"
                    )
        except Exception as e:
            # Log the exception details or handle them appropriately
            print(f"An error occurred: {e}")
            raise  # Re-raise the exception after handling it for further debugging or propagation
