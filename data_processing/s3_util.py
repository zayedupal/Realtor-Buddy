import os
import shutil
import zipfile

import boto3
from botocore.exceptions import NoCredentialsError


def download_and_extract_s3_zip(bucket_name, s3_key, download_path, extract_to):
    """
    This function can download a zipped file from S3 and extract the content to a local directory
    Args:
        bucket_name: real-estate-realtor-buddy
        s3_key: data.zip
        download_path: /tmp/data.zip
        extract_to: data

    Returns: None
    """
    s3 = boto3.client("s3")

    try:
        # Download the file from S3
        s3.download_file(bucket_name, s3_key, download_path)
        print(f"Downloaded {s3_key} from {bucket_name} to {download_path}")

        # Remove the existing extract directory if it exists
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
            print(f"Removed existing directory {extract_to}")

        # Create the extract directory
        os.makedirs(extract_to)

        # Extract the zip file
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {download_path} to {extract_to}")

    except NoCredentialsError:
        print("Credentials not available")


def zip_and_upload_data_directory(bucket_name, upload_key, directory_path, zip_path):
    """
    This function can zip any local directory and upload it to a desired location in S3
    In case we are modifying the content of our local data directory and want to upload it to S3, we can use this.

    Args:
        bucket_name: 'real-estate-realtor-buddy'
        upload_key: f'data-{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        directory_path: 'data'
        zip_path: f'/tmp/{upload_key}'

    Returns: None
    """
    s3 = boto3.client("s3")

    try:
        # Remove the existing zip file if it exists
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed existing file {zip_path}")

        # Zip the directory
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", directory_path)
        print(f"Zipped {directory_path} to {zip_path}")

        # Upload the zip file to S3
        s3.upload_file(zip_path, bucket_name, upload_key)
        print(f"Uploaded {zip_path} to s3://{bucket_name}/{upload_key}")

    except NoCredentialsError:
        print("Credentials not available")
