# Plan: HTTP Bootstrap for Vector Database

## Goal
Enable PyContextify to automatically download the vector database (FAISS index and metadata files) from an HTTP endpoint when the local cache is missing, so first-run deployments can hydrate themselves without manual file copies.

## Key Questions to Answer
- How should the bootstrap source be configured (environment variables, CLI options)?
- In what lifecycle phase should the download occur, and how do we coordinate it with the existing auto-load logic?
- What integrity and error handling safeguards are required?
- How do we avoid interfering with normal save/load operations once the files exist locally?

## Proposed Changes
1. **Configuration Enhancements**
   - Add new optional settings for the bootstrap URL (e.g., `PYCONTEXTIFY_INDEX_BOOTSTRAP_URL` and `--index-bootstrap-url`).
   - Allow specifying separate URLs for the FAISS index and metadata, or support a single archive with a new `--index-bootstrap-archive` option.
   - Extend `Config.get_index_paths` or a new helper to expose these values to the IndexManager.

2. **Bootstrap Orchestration in `IndexManager`**
   - In `_auto_load`, before checking for local files, determine whether bootstrap should run (URL provided and at least one target file missing).
   - Introduce a `_bootstrap_index_from_http` helper that downloads required assets into temporary files and atomically renames them into place.
   - Ensure concurrency safety by using filesystem locks or per-run temp directories to prevent partial writes if multiple processes start simultaneously.

3. **Download & Extraction Utilities**
   - Use `requests` (already a dependency) with streaming download to handle large index files.
   - Support both direct file downloads and archive formats (ZIP/TAR); use Python’s stdlib (`zipfile`, `tarfile`) for extraction.
   - Verify integrity via optional checksum configuration (`PYCONTEXTIFY_INDEX_BOOTSTRAP_SHA256`) and fail with a clear error if validation fails.

4. **Logging & Error Handling**
   - Emit informative logs for each phase (starting download, bytes transferred, extraction success).
   - On failure, fall back to current behavior (start with an empty index) but log at WARNING level so operators can intervene.
   - Consider retries with exponential backoff for transient network issues.

5. **Testing Strategy**
   - Unit tests for new configuration parsing logic.
   - Integration-style tests using a local HTTP server (e.g., `http.server`) to verify end-to-end bootstrap into a temporary directory.
   - Tests covering failure modes: missing URL, checksum mismatch, download errors, archive extraction problems.

6. **Documentation Updates**
   - Update README or a new deployment guide section describing the bootstrap feature and configuration options.
   - Provide examples of environment variable configuration for common platforms.

## Open Questions / Follow-Ups
- Should bootstrap also support HTTPS with client certificates or authentication headers out of the box?
- Do we need to support resumable downloads for very large indices?
- How should we coordinate bootstrap with the backup/restore mechanism already present in `VectorStore`?
