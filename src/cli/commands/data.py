import click
import requests
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

@click.group()
def data_group():
    """Commands for managing raw data."""
    pass

@data_group.command("download-docs")
@click.argument("url", type=str)  # Remove the help parameter - it's not allowed for arguments
@click.option(
    "--output-dir",
    "-o",
    default="data/raw",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Directory to save the downloaded document. Defaults to 'data/raw/'.",
)
@click.option(
    "--filename",
    "-n",
    type=str,
    default=None,
    help="Optional filename for the downloaded content. If not specified, inferred from URL or Content-Disposition header."
)
def download_docs(url: str, output_dir: str, filename: str | None):
    """
    Downloads a single file (e.g., PDF, HTML, etc.) from a given URL.

    The downloaded file will be saved as a monolithic file, without any processing
    like chunking or recursive downloading.

    Arguments:
        URL: The URL of the document to download (e.g., PDF, single HTML file).

    Example:
    \b
    techdoc-genie data download-docs https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf -o data/raw/postgresql
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Attempting to download from: {url}")
    click.echo(f"Saving to directory: {output_path}")

    # Infer filename if not provided
    if filename is None:
        parsed_url = urlparse(url)
        # Try to get filename from URL path first
        filename_from_url = Path(parsed_url.path).name
        if filename_from_url and not filename_from_url.endswith('/'):
            filename = filename_from_url
        else:
            # Fallback to checking Content-Disposition header
            try:
                with requests.head(url, allow_redirects=True, timeout=5) as head_response:
                    head_response.raise_for_status()
                    if 'Content-Disposition' in head_response.headers:
                        cd = head_response.headers['Content-Disposition']
                        # Regex to extract filename from Content-Disposition
                        fname_match = re.findall(r'filename\*?=(?:UTF-8\'\')?"?([^"]+)"?', cd)
                        if fname_match:
                            filename = unquote(fname_match[0])
                    
                    if not filename: # If still no filename, try from URL path suffix
                        # Example: https://example.com/some_file (no extension) -> downloaded_file
                        # Example: https://example.com/some_file.pdf -> downloaded_file.pdf
                        ext = Path(parsed_url.path).suffix
                        base_name = "downloaded_document"
                        if filename_from_url:
                            base_name = filename_from_url.split('.')[0] # Use part before extension if exists
                        filename = base_name + (ext if ext else '')

            except requests.exceptions.RequestException:
                # If HEAD request fails, fallback to a generic name
                filename = "downloaded_document" + Path(parsed_url.path).suffix # Try to keep original extension
                if not Path(parsed_url.path).suffix: # if no suffix, add .html as a common default
                    filename += ".html"
                
    if not filename: # Final fallback if all else fails
        filename = "downloaded_document.html" # Assume HTML as a common web document

    try:
        click.echo(f"Downloading '{filename}'...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        file_path = output_path / filename

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        click.echo(f"✅ Successfully downloaded to: {file_path}")
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Error downloading: {e}", err=True)
        raise click.Abort()


# import click
# import requests
# import re
# from pathlib import Path
# from urllib.parse import urlparse, unquote

# @click.group()
# def data_group():
#     """Commands for managing raw data."""
#     pass

# @data_group.command("download-docs")
# @click.argument("url", type=str)  # Remove the help parameter - it's not allowed for arguments
# @click.option(
#     "--output-dir",
#     "-o",
#     default="data/raw",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save the downloaded document. Defaults to 'data/raw/'.",
# )
# @click.option(
#     "--filename",
#     "-n",
#     type=str,
#     default=None,
#     help="Optional filename for the downloaded content. If not specified, inferred from URL or Content-Disposition header."
# )
# def download_docs(url: str, output_dir: str, filename: str | None):
#     """
#     Downloads a single file (e.g., PDF, HTML, etc.) from a given URL.

#     The downloaded file will be saved as a monolithic file, without any processing
#     like chunking or recursive downloading.

#     Arguments:
#         URL: The URL of the document to download (e.g., PDF, single HTML file).

#     Example:
#     \b
#     techdoc-genie data download-docs https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf -o data/raw/postgresql
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     click.echo(f"Attempting to download from: {url}")
#     click.echo(f"Saving to directory: {output_path}")

#     # Infer filename if not provided
#     if filename is None:
#         parsed_url = urlparse(url)
#         # Try to get filename from URL path first
#         filename_from_url = Path(parsed_url.path).name
#         if filename_from_url and not filename_from_url.endswith('/'):
#             filename = filename_from_url
#         else:
#             # Fallback to checking Content-Disposition header
#             try:
#                 with requests.head(url, allow_redirects=True, timeout=5) as head_response:
#                     head_response.raise_for_status()
#                     if 'Content-Disposition' in head_response.headers:
#                         cd = head_response.headers['Content-Disposition']
#                         # Regex to extract filename from Content-Disposition
#                         fname_match = re.findall(r'filename\*?=(?:UTF-8\'\')?"?([^"]+)"?', cd)
#                         if fname_match:
#                             filename = unquote(fname_match[0])
                    
#                     if not filename: # If still no filename, try from URL path suffix
#                         # Example: https://example.com/some_file (no extension) -> downloaded_file
#                         # Example: https://example.com/some_file.pdf -> downloaded_file.pdf
#                         ext = Path(parsed_url.path).suffix
#                         base_name = "downloaded_document"
#                         if filename_from_url:
#                             base_name = filename_from_url.split('.')[0] # Use part before extension if exists
#                         filename = base_name + (ext if ext else '')

#             except requests.exceptions.RequestException:
#                 # If HEAD request fails, fallback to a generic name
#                 filename = "downloaded_document" + Path(parsed_url.path).suffix # Try to keep original extension
#                 if not Path(parsed_url.path).suffix: # if no suffix, add .html as a common default
#                     filename += ".html"
                
#     if not filename: # Final fallback if all else fails
#         filename = "downloaded_document.html" # Assume HTML as a common web document

#     try:
#         click.echo(f"Downloading '{filename}'...")
#         response = requests.get(url, stream=True)
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

#         file_path = output_path / filename

#         with open(file_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         click.echo(f"✅ Successfully downloaded to: {file_path}")
#     except requests.exceptions.RequestException as e:
#         click.echo(f"❌ Error downloading: {e}", err=True)
#         raise click.Abort()


# import click
# import requests
# import re
# from pathlib import Path
# from urllib.parse import urlparse, unquote

# @click.group()
# def data_group():
#     """Commands for managing raw data."""
#     pass

# @data_group.command("download-docs")
# @click.argument("url", type=str, help="The URL of the document to download (e.g., PDF, single HTML file).")
# @click.option(
#     "--output-dir",
#     "-o",
#     default="data/raw",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save the downloaded document. Defaults to 'data/raw/'.",
# )
# @click.option(
#     "--filename",
#     "-n",
#     type=str,
#     default=None,
#     help="Optional filename for the downloaded content. If not specified, inferred from URL or Content-Disposition header."
# )
# def download_docs(url: str, output_dir: str, filename: str | None):
#     """
#     Downloads a single file (e.g., PDF, HTML, etc.) from a given URL.

#     The downloaded file will be saved as a monolithic file, without any processing
#     like chunking or recursive downloading.

#     Example:
#     \b
#     techdoc-genie data download-docs https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf -o data/raw/postgresql
#     techdoc-genie data download-docs https://www.postgresql.org/docs/16/ --format html-recursive -o data/raw/postgresql-html
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     click.echo(f"Attempting to download from: {url}")
#     click.echo(f"Saving to directory: {output_path}")

#     # Infer filename if not provided
#     if filename is None:
#         parsed_url = urlparse(url)
#         # Try to get filename from URL path first
#         filename_from_url = Path(parsed_url.path).name
#         if filename_from_url and not filename_from_url.endswith('/'):
#             filename = filename_from_url
#         else:
#             # Fallback to checking Content-Disposition header
#             try:
#                 with requests.head(url, allow_redirects=True, timeout=5) as head_response:
#                     head_response.raise_for_status()
#                     if 'Content-Disposition' in head_response.headers:
#                         cd = head_response.headers['Content-Disposition']
#                         # Regex to extract filename from Content-Disposition
#                         fname_match = re.findall(r'filename\*?=(?:UTF-8\'\')?"?([^"]+)"?', cd)
#                         if fname_match:
#                             filename = unquote(fname_match[0])
                    
#                     if not filename: # If still no filename, try from URL path suffix
#                         # Example: https://example.com/some_file (no extension) -> downloaded_file
#                         # Example: https://example.com/some_file.pdf -> downloaded_file.pdf
#                         ext = Path(parsed_url.path).suffix
#                         base_name = "downloaded_document"
#                         if filename_from_url:
#                             base_name = filename_from_url.split('.')[0] # Use part before extension if exists
#                         filename = base_name + (ext if ext else '')

#             except requests.exceptions.RequestException:
#                 # If HEAD request fails, fallback to a generic name
#                 filename = "downloaded_document" + Path(parsed_url.path).suffix # Try to keep original extension
#                 if not Path(parsed_url.path).suffix: # if no suffix, add .html as a common default
#                     filename += ".html"
                
#     if not filename: # Final fallback if all else fails
#         filename = "downloaded_document.html" # Assume HTML as a common web document

#     try:
#         click.echo(f"Downloading '{filename}'...")
#         response = requests.get(url, stream=True)
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

#         file_path = output_path / filename

#         with open(file_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         click.echo(f"✅ Successfully downloaded to: {file_path}")
#     except requests.exceptions.RequestException as e:
#         click.echo(f"❌ Error downloading: {e}", err=True)
#         raise click.Abort()