# Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import time
import boto3
import shutil
import tempfile

from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any, List

from tqdm import tqdm

CACHE_DIR = "/dockerx/Local/users/bangzheng"
class CustomTempDirectory:
    def __init__(self, base_dir: str=CACHE_DIR, prefix: str = 'tmp_'):
        """
        Create a temporary directory under specified base directory.
        
        Args:
            base_dir: Base directory path where temp directory will be created
            prefix: Prefix for the temp directory name (default: 'tmp_')
        """
        os.makedirs(base_dir, exist_ok=True)  # Ensure base directory exists
        self.name = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    
    def cleanup(self,checkpoint_name=None):
        """Remove the temporary directory and its contents"""

        if checkpoint_name:
            checkpoint_pth = os.path.join(self.name,checkpoint_name)
            if os.path.exists(checkpoint_pth):
                shutil.rmtree(checkpoint_pth, ignore_errors=True)
        else:
            if os.path.exists(self.name):
                shutil.rmtree(self.name, ignore_errors=True)
    
    def __enter__(self):
        """Support for 'with' statement"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting 'with' statement"""
        self.cleanup()

def create_temp_dir(base_path: str =  'checkpoints', prefix: str = 'tmp-model-') -> CustomTempDirectory:
    """
    Create a temporary directory under specified path.
    
    Args:
        base_path: Base directory path
        prefix: Prefix for temp directory name
    
    Returns:
        CustomTempDirectory object
    """
    return CustomTempDirectory(base_path, prefix)

class OCIFolderCheckpointHandler:
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str,
        bucket_name: str,
        region_name: str = "us-east-1",
        retries: int = 3,
        delay: int = 5
    ):
        """
        Initialize S3 client for OCI object storage with folder-based checkpoint handling.
        
        Args:
            access_key_id: OCI access key ID
            secret_access_key: OCI secret access key
            endpoint_url: OCI S3-compatible endpoint URL
            bucket_name: Name of the bucket to use
            region_name: Region name (default: us-east-1)
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=Config(signature_version='s3v4')
        )
        self.bucket_name = bucket_name
        self.retries = retries
        self.delay = delay

    def checkpoint_exists(self, checkpoint_folder: str) -> bool:
        """
        Check if a checkpoint folder exists in the S3 bucket.
        
        Args:
            checkpoint_folder: Path to the checkpoint folder in S3
            
        Returns:
            bool: True if checkpoint folder exists and contains files, False otherwise
        """
        files = self._get_folder_contents(checkpoint_folder)
        return len(files) > 0

    def _upload_directory_recursively(self, local_dir: str, s3_prefix: str) -> None:
        """
        Recursively upload a directory and its contents to S3.
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (path) to upload to
        """
        for root, dirs, files in os.walk(local_dir):
            # Calculate relative path from the base directory
            relative_path = os.path.relpath(root, local_dir)
            if relative_path == '.':
                relative_path = ''
            
            # Upload each file in the current directory
            for file in files:
                local_path = os.path.join(root, file)
                # Construct S3 key maintaining directory structure
                if relative_path:
                    s3_key = f"{s3_prefix}{relative_path}/{file}"
                else:
                    s3_key = f"{s3_prefix}{file}"
                
                # Normalize path separators for S3
                s3_key = s3_key.replace('\\', '/')

                # Upload file to S3 with retries
                try:
                    # Check if the file exists in the S3 bucket
                    self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    print(f"{local_path} already exists in S3. Skipping upload.")
                    continue
                except ClientError:
                    for attempt in range(self.retries):
                        try:
                            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                            print(f"{local_path} uploaded!")
                            break
                        except Exception as e:
                            if attempt < self.retries - 1:
                                print(f"Upload failed for {local_path}. Retrying in {self.delay} seconds...")
                                time.sleep(self.delay)
                            else:
                                print(f"Failed to upload {local_path} after {self.retries} attempts.")
                                raise e

    def save_checkpoint(
        self,
        temp_dir: CustomTempDirectory,
        checkpoint_folder: str,
    ) -> None:
        """
        Save model checkpoint as a folder to S3.
        
        Args:
            temp_dir: Temporary directory containing checkpoint files
            checkpoint_folder: Folder path in S3 to save the checkpoint
        """
        # Ensure checkpoint folder path ends with '/'
        checkpoint_folder = checkpoint_folder.rstrip('/') + '/'
        
        # Delete existing checkpoint if it exists
        # if self.checkpoint_exists(checkpoint_folder):
        #     self.delete_checkpoint(checkpoint_folder)

        if isinstance(temp_dir,CustomTempDirectory):
            local_folder = temp_dir.name
        else:
            local_folder = temp_dir
        # Recursively upload the entire directory
        self._upload_directory_recursively(local_folder, checkpoint_folder)

    def _get_folder_contents(self, folder_path: str) -> List[str]:
        """
        Get list of all files in a folder in S3 recursively.
        
        Args:
            folder_path: Path to the folder in S3
            
        Returns:
            List of file paths in the folder and subfolders
        """
        folder_path = folder_path.rstrip('/') + '/'
        
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        # Use paginator to handle cases with many files
        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=folder_path
        ):
            if 'Contents' in page:
                files.extend([obj['Key'] for obj in page['Contents']])
        
        return files


    def get_max_checkpoint_iter(self, checkpoint_folder: str, files: List[str])->int:
        iters = set()
        for file_path in files:
            # Get relative path from checkpoint folder
            relative_path = file_path[len(checkpoint_folder):].lstrip('/')
            folder_name = relative_path.split('/')[0]
            # print(files)
            if folder_name.startswith('checkpoint'):
                iteration = int(folder_name.split('-')[1])
                iters.add(iteration)
        iters = list(iters)
        return max(iters) if len(iters) else 0
    
    def get_num_shards_of_checkpoint(self, checkpoint_folder: str)->int:
        r"""
        Get the number of shards a model has been split into.
        """
        # Get all files from checkpoint folder
        files = self._get_folder_contents(checkpoint_folder)
        max_checkpoint_iter = self.get_max_checkpoint_iter( checkpoint_folder, files)
        # get list of files in max_iter's ckpt
        # TODO: Needs to redine the logic (based on bf16 shards or something else)
        files = self._get_folder_contents(os.path.join(checkpoint_folder.rstrip('/'), f'checkpoint-{max_checkpoint_iter}/global_step{max_checkpoint_iter}/'))
        
        files = [f for f in files if f.split("/")[-1].startswith('bf16_zero_pp_rank')]
        return len(files)
        

    def load_checkpoint(
        self,
        checkpoint_folder: str, 
        temp_dir: CustomTempDirectory,
        inference_mode = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint from a folder in S3.
        
        Args:
            checkpoint_folder: Folder path in S3 containing the checkpoint
            temp_dir: Temporary directory to download files to
            
        Returns:
            Temporary directory containing downloaded files
        """
        if temp_dir is None or not os.path.exists(temp_dir.name):
            temp_dir = create_temp_dir()

        # Get all files from checkpoint folder
        files = self._get_folder_contents(checkpoint_folder)
        max_checkpoint_iter = self.get_max_checkpoint_iter( checkpoint_folder, files)

        for file_path in tqdm(files,desc="downloading from oci"):
            # Get relative path from checkpoint folder
            relative_path = file_path[len(checkpoint_folder):].lstrip('/')
            checkpoint_dir = f'checkpoint-{max_checkpoint_iter}/'
            if relative_path.startswith('checkpoint') and not relative_path.startswith(checkpoint_dir) or relative_path.startswith('.'):
                continue
            if inference_mode and relative_path.startswith('global_step'):
                print(f"Skipped {relative_path}")
                continue
            local_path = os.path.join(temp_dir.name, relative_path)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            try:
                self.s3_client.download_file(self.bucket_name, file_path, local_path)
                print(f"Downloaded {self.bucket_name}/{file_path} to {local_path}")
            except:
                print(relative_path)

        return temp_dir

    def list_checkpoints(self, base_folder: str = '') -> List[str]:
        """
        List all checkpoint folders in the given base folder.
        
        Args:
            base_folder: Optional base folder to search in
            
        Returns:
            List of checkpoint folder paths
        """
        base_folder = base_folder.rstrip('/') + '/' if base_folder else ''
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=base_folder,
            Delimiter='/'
        )
        
        checkpoints = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                checkpoints.append(prefix['Prefix'])
        
        return checkpoints

    def delete_checkpoint(self, checkpoint_folder: str) -> None:
        """
        Delete a checkpoint folder and all its contents from S3.
        
        Args:
            checkpoint_folder: Folder path to delete
        """
        files = self._get_folder_contents(checkpoint_folder)
        
        # Delete all files in the folder
        for file_path in files:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )

if __name__ == "__main__":