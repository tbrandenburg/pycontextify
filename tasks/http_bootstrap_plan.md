# Plan: HTTP Bootstrap for Vector Database

## Goal
Enable PyContextify to automatically download the vector database (FAISS index and metadata files) from an HTTP-served archive when the local cache is missing, so first-run deployments can hydrate themselves without manual file copies while guaranteeing that already bootstrapped local files remain untouched.

## Key Questions to Answer
- How should the bootstrap source be configured (environment variables, CLI options)?
- In what lifecycle phase should the download occur, and how do we coordinate it with the existing auto-load logic?
- What integrity and error handling safeguards are required?
- How do we avoid interfering with normal save/load operations once the files exist locally?

## Proposed Changes
1. **Configuration Enhancements**
   - Add a new optional setting for the bootstrap archive URL (e.g., `PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL` and `--index-bootstrap-archive-url`). The archive must be served over HTTP/HTTPS (`https://`) or exposed via a file URI (`file://`) and is expected to be either a ZIP file or a TAR-GZ archive that already contains both the FAISS index and metadata artifacts.
   - Assume a checksum file (`.sha256`) lives alongside the archive and shares the same filename stem. When the archive URL is provided, derive the checksum URL by appending `.sha256` so operators only need to supply the archive location.
   - Extend `Config.get_index_paths` or add a helper so that `IndexManager` can access both the archive URL and the derived checksum URL.

2. **Bootstrap Orchestration in `IndexManager`**
   - In `_auto_load`, detect whether bootstrap should run: bootstrap only when at least one expected local artifact is missing and an archive URL is configured. If both FAISS and metadata files already exist locally, skip bootstrap entirely to avoid overwriting previously hydrated data.
   - Keep server startup fast by launching bootstrap work in the background: `_auto_load` should schedule the download/extract routine (e.g., via a background thread or asynchronous task) and then return immediately so MCP server startup is not blocked. The bootstrap runs eagerly once scheduled—without waiting for the first index access—but it executes off the main startup path to maintain responsiveness.
   - Introduce a `_bootstrap_index_from_http` helper that downloads the archive and checksum into a dedicated temporary directory, verifies integrity, extracts the contents, and then atomically moves only the missing artifacts into place.
   - Before invoking any network download, check for `VectorStore`-managed backups (`*_backup_*` files) that correspond to missing artifacts. When a recent backup exists, restore it in place and skip the download entirely so bootstrap defers to the existing backup/restore workflow.
   - Ensure concurrency safety by holding a filesystem lock or using per-run temp directories plus `os.replace` when moving files so that partially extracted data never overwrites existing files, even under parallel startups. The background worker should respect the same locking so that simultaneous processes do not duplicate work or trample each other.

3. **Download, Verification & Extraction Utilities**
   - Use `requests` (already a dependency) with streaming download to handle large archives, writing them to temporary paths before any rename operation so existing files are never truncated mid-run.
   - Fetch the checksum file derived from the archive URL and validate the archive contents before extraction. Fail fast with a clear error if validation does not match, leaving the local directory untouched.
   - Support both ZIP and TAR-GZ archives using Python’s stdlib (`zipfile`, `tarfile`). Extract into a temporary staging directory and, after extraction, move each expected artifact into place only if its destination file is absent. Log a warning if the archive omits any expected files.
   - Resumable or chunked downloads are explicitly out of scope for the first iteration; a single streaming download is sufficient.

4. **Logging & Error Handling**
   - Emit informative logs for each phase (starting download, checksum verification, extraction success, and explicit messages when bootstrap is skipped because files already exist).
   - On failure, fall back to current behavior (start with an empty index) but log at WARNING level so operators can intervene.
   - Consider retries with exponential backoff for transient network issues.

5. **Testing Strategy**
   - Unit tests for new configuration parsing logic, including checksum URL derivation.
   - Integration-style tests using a local HTTP server (e.g., `http.server`) to verify end-to-end bootstrap into a temporary directory with ZIP and TAR-GZ archives.
   - Tests covering failure modes: missing archive URL, checksum mismatch, download errors, archive extraction problems, and archives missing expected files.

6. **Documentation Updates**
   - Update README or a new deployment guide section describing the bootstrap feature and configuration options.
   - Provide examples of environment variable configuration for common platforms and note the required archive layout and checksum co-location.

## Open Questions / Follow-Ups
- Should bootstrap also support HTTPS with client certificates or authentication headers out of the box?
- After the initial release, should we revisit resumable downloads for very large archives as a follow-up enhancement?
- Does the "restore from the newest `VectorStore` backup before downloading" strategy need additional safeguards (e.g., backup freshness checks)?
