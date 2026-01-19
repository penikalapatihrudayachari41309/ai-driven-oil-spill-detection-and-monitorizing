import gdown

# Use gdown to download files from Google Drive

def download_file_from_drive(file_url, output_path):
    gdown.download(file_url, output_path, quiet=False)