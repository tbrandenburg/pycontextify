# HTTP Bootstrap for PyContextify

## Overview

The HTTP Bootstrap feature allows PyContextify to automatically download and initialize its vector database index from a pre-built archive hosted on HTTP/HTTPS or available on the local filesystem. This is ideal for:

- **Production deployments** where you want to ship a pre-built index
- **Cold-start scenarios** where rebuilding the index would be too slow
- **Multi-instance deployments** where all instances share the same initial index
- **CI/CD pipelines** where index artifacts are built once and distributed

## How It Works

1. **On Startup**: If configured, PyContextify checks for missing index files
2. **Backup Check**: First attempts to restore from local backups if available
3. **Bootstrap**: If files are still missing, downloads and extracts the archive in the background
4. **Non-Blocking**: The server starts immediately; bootstrap happens asynchronously
5. **Idempotent**: If index files already exist, bootstrap is skipped entirely

## Configuration

### Environment Variables

```bash
# Required: URL to the bootstrap archive (ZIP, TAR-GZ, or TGZ)
PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL="https://example.com/index_bootstrap.tar.gz"

# Optional: URL to the SHA256 checksum file (auto-derived if not provided)
# Will automatically append .sha256 to archive URL if not specified
PYCONTEXTIFY_INDEX_BOOTSTRAP_CHECKSUM_URL="https://example.com/index_bootstrap.tar.gz.sha256"

# Standard index configuration
PYCONTEXTIFY_INDEX_DIR="./index_data"
PYCONTEXTIFY_INDEX_NAME="semantic_index"
PYCONTEXTIFY_AUTO_LOAD="true"
```

### CLI Arguments

```bash
# Start with bootstrap from HTTPS
pycontextify --index-bootstrap-archive-url https://example.com/index.tar.gz

# Start with bootstrap from local file
pycontextify --index-bootstrap-archive-url file:///path/to/index.zip

# Combine with other options
pycontextify \
  --index-path ./my_index \
  --index-name production_index \
  --index-bootstrap-archive-url https://cdn.example.com/index_v1.2.3.tar.gz
```

## Archive Structure

### Required Files

The bootstrap archive **must** contain these files:

```
archive.tar.gz/
├── semantic_index.faiss       # FAISS vector index (binary)
└── semantic_index.pkl          # Metadata store (pickle)
```

**Important:**
- File names must match your `PYCONTEXTIFY_INDEX_NAME` setting (default: `semantic_index`)
- Files can be at any depth in the archive (searched recursively)
- Archive can include other files, but only `.faiss` and `.pkl` are used

### Example with Default Settings

If using default settings:
- `PYCONTEXTIFY_INDEX_NAME="semantic_index"` (default)
- `PYCONTEXTIFY_INDEX_DIR="./index_data"` (default)

Your archive should contain:
- `semantic_index.faiss`
- `semantic_index.pkl`

### Example with Custom Name

If using:
- `PYCONTEXTIFY_INDEX_NAME="production_search"`

Your archive should contain:
- `production_search.faiss`
- `production_search.pkl`

## Creating Bootstrap Archives

### Step 1: Build Your Index Locally

```bash
# Index your content normally
pycontextify
# Then use MCP functions to index content...
```

This creates files in `./index_data/`:
- `semantic_index.faiss`
- `semantic_index.pkl`

### Step 2: Create Archive

#### Option A: ZIP Archive

```bash
cd index_data
zip -r ../index_bootstrap.zip semantic_index.faiss semantic_index.pkl
cd ..
```

#### Option B: TAR-GZ Archive (Recommended)

```bash
tar -czf index_bootstrap.tar.gz -C index_data semantic_index.faiss semantic_index.pkl
```

### Step 3: Generate Checksum

#### Linux/macOS:
```bash
# Using shasum
shasum -a 256 index_bootstrap.tar.gz > index_bootstrap.tar.gz.sha256

# Or using sha256sum
sha256sum index_bootstrap.tar.gz > index_bootstrap.tar.gz.sha256
```

#### Windows (PowerShell):
```powershell
(Get-FileHash index_bootstrap.tar.gz -Algorithm SHA256).Hash.ToLower() | 
  Out-File -Encoding ascii index_bootstrap.tar.gz.sha256
```

#### Python:
```python
import hashlib
from pathlib import Path

archive_path = Path("index_bootstrap.tar.gz")
hash_obj = hashlib.sha256()

with archive_path.open("rb") as f:
    while chunk := f.read(1024 * 1024):
        hash_obj.update(chunk)

checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
checksum_path.write_text(hash_obj.hexdigest())
```

### Checksum File Formats

Both formats are supported:

```bash
# Format 1: Hash only
a1b2c3d4e5f6...  

# Format 2: Hash with filename
a1b2c3d4e5f6... index_bootstrap.tar.gz
```

## Hosting Options

### Option 1: Static Web Server

```bash
# Simple HTTP server for testing
python -m http.server 8000 --directory /path/to/archives

# Access at:
# http://localhost:8000/index_bootstrap.tar.gz
```

### Option 2: Cloud Storage

```bash
# AWS S3
aws s3 cp index_bootstrap.tar.gz s3://my-bucket/ --acl public-read
aws s3 cp index_bootstrap.tar.gz.sha256 s3://my-bucket/ --acl public-read
# URL: https://my-bucket.s3.amazonaws.com/index_bootstrap.tar.gz

# Google Cloud Storage
gsutil cp index_bootstrap.tar.gz gs://my-bucket/
gsutil cp index_bootstrap.tar.gz.sha256 gs://my-bucket/
gsutil acl ch -u AllUsers:R gs://my-bucket/index_bootstrap*
# URL: https://storage.googleapis.com/my-bucket/index_bootstrap.tar.gz

# Azure Blob Storage
az storage blob upload --container-name my-container \
  --file index_bootstrap.tar.gz --name index_bootstrap.tar.gz
```

### Option 3: Local Filesystem

```bash
# Use file:// URLs for local archives
ARCHIVE_PATH="/path/to/index_bootstrap.tar.gz"

# On Windows
file:///C:/path/to/index_bootstrap.tar.gz

# On Linux/macOS  
file:///path/to/index_bootstrap.tar.gz
```

## Retry Behavior

The bootstrap implementation includes intelligent retry logic:

### Transient Errors (Automatic Retry)
- **Timeouts**: Connection or read timeouts
- **Connection Errors**: Network unavailable, DNS failures
- **HTTP 408**: Request Timeout
- **HTTP 429**: Too Many Requests
- **HTTP 5xx**: Server errors

### Retry Strategy
- **Maximum Attempts**: 3
- **Backoff**: Exponential (1s, 2s, 4s)
- **Atomic Writes**: Downloads to `.tmp` file first, then atomic rename

### Non-Retriable Errors (Immediate Failure)
- **HTTP 4xx** (except 408, 429): Client errors like 403, 404
- **Invalid checksums**: Archive integrity check failed
- **Malformed archives**: Cannot extract or missing required files

## Background Execution

Bootstrap runs in a **background daemon thread** to ensure:
- ✅ **Fast Startup**: Server is ready immediately
- ✅ **Non-Blocking**: MCP functions work while bootstrap runs
- ✅ **Automatic**: No manual intervention needed
- ✅ **Safe**: Uses filesystem locks to prevent concurrent downloads

## Backup Integration

Bootstrap intelligently integrates with the backup system:

1. **Check for Missing Files**: Only downloads if index files don't exist
2. **Restore from Backup**: If `VectorStore` backups exist, uses them instead
3. **Skip Download**: If backup restoration succeeds, network download is skipped
4. **Fallback Chain**: Existing files → Backups → HTTP Bootstrap → Fresh start

## Security Considerations

### Checksum Verification

**Always use checksums in production!** The bootstrap process:
- ✅ Downloads checksum file (`.sha256`)
- ✅ Computes SHA256 of downloaded archive
- ✅ Compares checksums (case-insensitive)
- ❌ **Fails immediately** on mismatch

### HTTPS Requirement

- **Production**: Always use `https://` URLs
- **Development**: `http://` allowed for localhost only
- **File URLs**: `file://` supported for local testing

### Best Practices

```bash
# ✅ GOOD: HTTPS with checksum
https://cdn.example.com/index_v1.0.0.tar.gz
https://cdn.example.com/index_v1.0.0.tar.gz.sha256

# ✅ GOOD: Local file for testing
file:///tmp/index_bootstrap.zip
file:///tmp/index_bootstrap.zip.sha256

# ⚠️ AVOID: HTTP in production
http://insecure.example.com/index.tar.gz

# ❌ BAD: Missing checksum file
https://example.com/index.tar.gz
# (no .sha256 file)
```

## Troubleshooting

### Bootstrap Not Running

**Check logs for:**
```
Bootstrap archive URL not configured; skipping bootstrap
```

**Solution:** Set `PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL`

### Files Already Exist

```
Bootstrap skipped because index artifacts already exist
```

**This is normal!** Bootstrap only runs when files are missing. To force re-bootstrap:
```bash
rm index_data/semantic_index.faiss
rm index_data/semantic_index.pkl
```

### Checksum Mismatch

```
Bootstrap archive checksum mismatch: expected abc123..., got def456...
```

**Causes:**
- Archive file was modified or corrupted
- Checksum file doesn't match archive
- Network transmission error

**Solution:** Re-generate checksum or re-upload archive

### Download Failures

```
Failed to download https://example.com/index.tar.gz after 3 attempts
```

**Causes:**
- Network connectivity issues
- URL is incorrect or inaccessible
- Server is down

**Solution:** Check URL, network, and server availability

### Missing Files in Archive

```
Bootstrap archive missing expected index file semantic_index.faiss
```

**Causes:**
- Archive doesn't contain required `.faiss` file
- File name doesn't match `PYCONTEXTIFY_INDEX_NAME`

**Solution:** Verify archive contents and file names

### Concurrent Bootstrap

The bootstrap uses a **threading lock** (`_bootstrap_lock`) to ensure:
- Only one download per process
- Multiple concurrent startups are safe
- No duplicate network requests

## Example Deployment Workflows

### Docker Container

```dockerfile
FROM python:3.11

# Install PyContextify
RUN pip install pycontextify

# Configure bootstrap
ENV PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL=https://cdn.example.com/index.tar.gz
ENV PYCONTEXTIFY_INDEX_DIR=/app/index_data

# Start server (will auto-bootstrap)
CMD ["pycontextify"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pycontextify-config
data:
  bootstrap_url: "https://storage.googleapis.com/my-bucket/index_v1.tar.gz"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pycontextify
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: pycontextify
        image: pycontextify:latest
        env:
        - name: PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL
          valueFrom:
            configMapKeyRef:
              name: pycontextify-config
              key: bootstrap_url
```

### CI/CD Pipeline

```yaml
# GitHub Actions example
name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Build Index
        run: |
          # Build your index
          pycontextify --initial-documents docs/**/*.md
          
      - name: Create Bootstrap Archive
        run: |
          tar -czf index.tar.gz -C index_data semantic_index.faiss semantic_index.pkl
          sha256sum index.tar.gz > index.tar.gz.sha256
          
      - name: Upload to CDN
        run: |
          aws s3 cp index.tar.gz s3://my-cdn/index_${{ github.sha }}.tar.gz
          aws s3 cp index.tar.gz.sha256 s3://my-cdn/index_${{ github.sha }}.tar.gz.sha256
          
      - name: Deploy
        run: |
          export BOOTSTRAP_URL="https://my-cdn.s3.amazonaws.com/index_${{ github.sha }}.tar.gz"
          kubectl set env deployment/pycontextify \
            PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL=$BOOTSTRAP_URL
```

## Performance Characteristics

- **Background Download**: Server startup ~1-3 seconds
- **Download Time**: Depends on archive size and network speed
- **Typical Archive Sizes**:
  - Small (1K documents): 50-200 MB
  - Medium (10K documents): 200-500 MB
  - Large (100K+ documents): 500 MB - 2 GB
- **Extraction Time**: ~1-5 seconds for most archives
- **First Query**: Available immediately (using fresh index) or after bootstrap completes

## Monitoring

Enable debug logging to monitor bootstrap progress:

```bash
# Enable verbose logging
pycontextify --verbose

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Log messages to watch for:**
```
Scheduling bootstrap download for missing index artifacts
Downloading bootstrap archive from https://... (attempt 1/3)
Download completed successfully
Fetched checksum: abc123def456...
Bootstrap archive applied successfully; loading index
```

## FAQ

### Q: Can I update the bootstrap archive without restarting?
A: No, bootstrap only runs on startup. To use a new archive, restart PyContextify.

### Q: What happens if the download fails?
A: The server starts with an empty index. You can manually index content or fix the bootstrap URL and restart.

### Q: Can I use multiple archives?
A: No, only one bootstrap archive URL is supported. Combine all necessary files into a single archive.

### Q: Does bootstrap work with relationships.pkl?
A: Currently, only `.faiss` and `.pkl` files are required. The `_relationships.pkl` file is optional and will be recreated if missing.

### Q: Can I bootstrap from private S3/GCS buckets?
A: Yes, generate pre-signed URLs or use bucket permissions to make files accessible via HTTPS.

### Q: What if my index files exist but are corrupted?
A: Delete the corrupted files and restart to trigger re-bootstrap.

### Q: How do I version my bootstrap archives?
A: Include version in filename: `index_v1.2.3.tar.gz`. Update the URL when deploying new versions.

---

For more information, see the main [README.md](../README.md) and the [HTTP Bootstrap Plan](../tasks/http_bootstrap_plan.md).
