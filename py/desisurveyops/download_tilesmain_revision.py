import subprocess
import sys
import os
import argparse
import getpass

import subprocess
import re

def get_latest_revision(svn_url, username, password):
    """
    Get the most recent revision number for an SVN repository.
    
    Args:
        svn_url: URL to the SVN repository or file
        username: SVN username
        password: SVN password
    
    Returns:
        Latest revision number as an integer, or None if lookup fails
    """
    
    cmd = [
        'svn', 'info',
        '--show-item', 'revision',
        '--username', username,
        '--password', password,
        '--no-auth-cache',
        svn_url
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        revision = result.stdout.strip()
        return int(revision)
        
    except subprocess.CalledProcessError as e:
        print(f"Error getting latest revision: {e.stderr}")
        return None
    except ValueError:
        print(f"Error: Unexpected output from svn info: {result.stdout}")
        return None
    except FileNotFoundError:
        print("Error: SVN command-line client not found. Please install SVN.")
        return None

    
def get_revision_date(svn_url, revision, username, password):
    """
    Get the commit date for a specific SVN revision.
    
    Args:
        svn_url: URL to the SVN repository or file
        revision: The revision number to look up
        username: SVN username
        password: SVN password
    
    Returns:
        Date string in YYYYMMDD format, or None if lookup fails
    """
    
    cmd = [
        'svn', 'log',
        '--revision', str(revision),
        '--xml',
        '--username', username,
        '--password', password,
        '--no-auth-cache',
        svn_url
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # XML output contains date like: <date>2024-03-15T14:32:01.123456Z</date>
        match = re.search(r'<date>(\d{4})-(\d{2})-(\d{2})T', result.stdout)
        
        if match:
            year, month, day = match.groups()
            return f"{year}{month}{day}"
        else:
            print("Error: Could not parse date from SVN log output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error getting revision info: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: SVN command-line client not found. Please install SVN.")
        return None

    
def download_svn_file(svn_url, revision, revdate, outdir, username, password):
    """
    Download a specific revision of a file from an SVN server.
    
    Args:
        svn_url: Full URL to the file in SVN
        revision: The revision number to download
        revdate: The date of the revision in YYYYMMDD
        outdir: Local path to directory where the file will be saved
        username: SVN username
        password: SVN password
    """
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    output_path = os.path.join(outdir, f'tiles-main-{revdate}-rev{str(revision).zfill(5)}.ecsv')
    cmd = [
        'svn', 'export',
        '--revision', str(revision),
        '--force',
        '--username', username,
        '--password', password,
        '--no-auth-cache',
        svn_url,
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully downloaded revision {revision} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: SVN command-line client not found. Please install SVN.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download a specific revision of a file from an SVN server.'
    )
    parser.add_argument(
        '-r', '--revision',
        type=str,
        default='latest',
        help='Revision number to download'
    )
    
    parser.add_argument(
        '-s', '--svn-url',
        default='https://desi.lbl.gov/svn/data/surveyops/trunk/ops/tiles-main.ecsv',
        help='Full URL to the file in SVN (e.g., https://svn.example.com/repo/trunk/file.txt)'
    )
    
    parser.add_argument(
        '-o', '--outdir',
        default='./',
        help='Output directory including filename (e.g., ./downloads/myfile.txt)'
    )
    
    parser.add_argument(
        '-u', '--username',
        help='SVN username (will prompt if not provided)'
    )
    
    args = parser.parse_args()
    
    # Get credentials
    if args.username:
        username = args.username
    else:
        username = input('SVN Username: ')
    
    password = getpass.getpass('SVN Password: ')

    if args.revision.lower() == 'latest':
        revision = int(get_latest_revision(args.svn_url, username, password))
    else:
        revision = int(args.revision)
        
    revdate = get_revision_date(args.svn_url, revision, username, password)

    print(f"Downloading tiles-main.ecsv for {revision=} on revision date={revdate}")
    success = download_svn_file(
        svn_url=args.svn_url,
        revision=revision,
        outdir=args.outdir,
        revdate=revdate,
        username=username,
        password=password
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
