---
id: PROB-15
title: "Hardcoded localhost:8000 in Frontend Preventing LAN/Mobile Testing"
stage: not_solved
status: solved
created: 2026-11-01
updated: 2026-11-01 12:18
severity: Low
category: Frontend Network & Portability
---

# PROB-15: Hardcoded localhost:8000 in Frontend Preventing LAN/Mobile Testing

## 1. Problem Description
When testing SETU on a phone or secondary computer over local Wi-Fi (`http://192.168.x.x:5173`), API calls fail because `apiService.ts` and `AlphabetModal.tsx` hardcode `http://localhost:8000` rather than dynamically adapting to the host hostname or environment configuration.

## 2. Root Cause Analysis
1. In `frontend/src/services/apiService.ts` (line 62):
   ```typescript
   private static baseUrl = 'http://localhost:8000';
   ```
2. In `frontend/src/components/AlphabetModal.tsx` (line 64):
   ```typescript
   <WebcamAlphabetRecognition apiBaseUrl="http://localhost:8000" />
   ```
When opening the web application on a mobile device, `localhost` resolves to the mobile device itself (where no Python backend is running), causing API connections to fail.

## 3. Implementation Plan & Proposed Fix
1. In `frontend/src/services/apiService.ts`:
   - Initialize `baseUrl` dynamically:
     ```typescript
     private static baseUrl =
       import.meta.env.VITE_API_BASE_URL ||
       (typeof window !== 'undefined'
         ? `${window.location.protocol}//${window.location.hostname}:8000`
         : 'http://localhost:8000');
     ```
2. In `frontend/src/components/AlphabetModal.tsx`:
   - Use `ApiService.getBaseUrl()` instead of hardcoded `'http://localhost:8000'`.

## 4. Acceptance Criteria & Verification
- [x] Frontend dynamically constructs base URL from `window.location.hostname`.
- [x] TypeScript build succeeds.
