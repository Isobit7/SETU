---
id: PROB-14
title: "Windows PermissionError on Aborted Video Uploads in video.py"
stage: not_solved
status: solved
created: 2026-11-01
updated: 2026-11-01 12:18
severity: Medium
category: Backend API & File System Handling
---

# PROB-14: Windows PermissionError on Aborted Video Uploads in video.py

## 1. Problem Description
When a user uploads a video file that exceeds the 50MB size limit or when an upload is aborted midway, the backend throws an unhandled `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`, and leaves orphan temporary video files in `%TEMP%`.

## 2. Root Cause Analysis
In `backend/app/api/endpoints/video.py` (lines 39–53 and 81–85):
```python
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
temp_path = Path(temp_file.name)
try:
    while chunk := await file.read(1024 * 1024):
        bytes_read += len(chunk)
        if bytes_read > MAX_BYTES:
            raise HTTPException(status_code=413, detail="...")  # <--- RAISES
        temp_file.write(chunk)
    temp_file.close()  # <--- ONLY REACHED IF NO EXCEPTION
...
finally:
    if temp_path.exists():
        os.remove(temp_path)  # <--- FAILS ON WINDOWS IF FILE STILL OPEN
```
On Windows, a file cannot be deleted while any open handle to it exists. When `HTTPException` is raised during chunk streaming, `temp_file.close()` is skipped. The `finally:` block immediately calls `os.remove(temp_path)` on the open file, causing Windows to reject the deletion with `WinError 32`.

## 3. Implementation Plan & Proposed Fix
1. Ensure `temp_file.close()` is placed in its own nested `finally` block or safely closed before `os.remove()`:
   ```python
   finally:
       try:
           if not temp_file.closed:
               temp_file.close()
       except Exception:
           pass
       try:
           if temp_path.exists():
               os.remove(temp_path)
       except Exception as e:
           logger.warning("Could not delete temporary video file %s: %s", temp_path, e)
   ```

## 4. Acceptance Criteria & Verification
- [x] Exceeding 50MB upload limit returns clean HTTP 413 without `WinError 32`.
- [x] Temporary files are always removed cleanly from disk.
