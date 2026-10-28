/**
 * SETU — RealSign 26-Class Hand Landmark Preprocessor & Real-Time Framing Analyzer
 *
 * Provides:
 * 1. Robust 63-D rotation- and scale-invariant normalization with singularity guards.
 * 2. Proper wrist-centered sagittal reflection (avoiding image-space 1.0-x offset distortion).
 * 3. Real-time hand framing, distance, and positioning analysis for UI guidance.
 */

export interface Landmark3D {
  x: number;
  y: number;
  z?: number;
}

export interface HandFramingAnalysis {
  palmSpan: number;
  boxWidth: number;
  boxHeight: number;
  centerX: number;
  centerY: number;
  distanceStatus: 'too_far' | 'too_close' | 'optimal';
  framingStatus: 'off_edge' | 'optimal' | 'reposition';
  isIdeal: boolean;
  guidanceMessage: string;
}

/**
 * Normalizes 21 3D hand landmarks into a 63-D rotation- and scale-invariant vector.
 *
 * @param landmarks - Array of 21 hand landmarks ({x, y, z})
 * @param mirrorSagittal - Whether to mirror X coordinates across wrist (for left-hand signing)
 * @returns 63-element normalized float array
 */
export function normalizeHandLandmarks63D(
  landmarks: Landmark3D[],
  mirrorSagittal: boolean = false
): number[] {
  if (!landmarks || landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  // 1. Center on Wrist (landmark 0)
  const wrist = landmarks[0];
  const wx = wrist.x || 0;
  const wy = wrist.y || 0;
  const wz = wrist.z || 0;

  const translated = landmarks.map((p) => {
    let dx = (p.x || 0) - wx;
    let dy = (p.y || 0) - wy;
    let dz = (p.z || 0) - wz;
    if (mirrorSagittal) {
      dx = -dx;
    }
    return { x: dx, y: dy, z: dz };
  });

  // 2. Scale normalize using Euclidean distance from Wrist (0) to Middle MCP (9)
  const vWristMiddle = translated[9];
  let scale = Math.hypot(vWristMiddle.x, vWristMiddle.y, vWristMiddle.z);
  if (scale < 1e-4) scale = 1.0;

  const scaled = translated.map((p) => ({
    x: p.x / scale,
    y: p.y / scale,
    z: p.z / scale,
  }));

  // 3. In-plane rotation around Z axis so (x, y) vector of Middle MCP aligns with (0, -1)
  const vx = scaled[9].x;
  const vy = scaled[9].y;
  const normXY = Math.hypot(vx, vy);

  let rotated = scaled;
  // Guard against near-zero singularity when fingers point towards camera
  if (normXY > 0.02) {
    const theta = Math.atan2(-vx, -vy);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    rotated = scaled.map((p) => ({
      x: p.x * cosT - p.y * sinT,
      y: p.x * sinT + p.y * cosT,
      z: p.z,
    }));
  }

  // 4. Out-of-plane rotation around X axis so (y', z') of Middle MCP aligns with (-1, 0)
  const y9 = rotated[9].y;
  const z9 = rotated[9].z;
  const normYZ = Math.hypot(y9, z9);

  if (normYZ > 0.02 && Math.abs(z9) > 0.02) {
    const phi = Math.atan2(z9, -y9);
    const cosP = Math.cos(phi);
    const sinP = Math.sin(phi);
    rotated = rotated.map((p) => ({
      x: p.x,
      y: p.y * cosP - p.z * sinP,
      z: p.y * sinP + p.z * cosP,
    }));
  }

  // Exact canonical alignment for origin and middle MCP
  rotated[0] = { x: 0.0, y: 0.0, z: 0.0 };
  rotated[9] = { x: 0.0, y: -1.0, z: 0.0 };

  // 5. Flatten into 63-D array
  const features = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    features[i * 3 + 0] = rotated[i].x;
    features[i * 3 + 1] = rotated[i].y;
    features[i * 3 + 2] = rotated[i].z;
  }

  return Array.from(features);
}

/**
 * Evaluates hand framing, camera distance, and bounding box metrics in real-time.
 * Used to guide the user into the optimal detection zone.
 */
export function analyzeHandFraming(landmarks: Landmark3D[]): HandFramingAnalysis {
  if (!landmarks || landmarks.length !== 21) {
    return {
      palmSpan: 0,
      boxWidth: 0,
      boxHeight: 0,
      centerX: 0.5,
      centerY: 0.5,
      distanceStatus: 'too_far',
      framingStatus: 'off_edge',
      isIdeal: false,
      guidanceMessage: 'Position hand inside camera view',
    };
  }

  let minX = 1.0;
  let maxX = 0.0;
  let minY = 1.0;
  let maxY = 0.0;

  for (let i = 0; i < 21; i++) {
    const x = landmarks[i].x || 0;
    const y = landmarks[i].y || 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const boxWidth = Math.max(0.001, maxX - minX);
  const boxHeight = Math.max(0.001, maxY - minY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  // Palm span: distance between Wrist (0) and Middle MCP (9)
  const w = landmarks[0];
  const m = landmarks[9];
  const palmSpan = Math.hypot((m.x || 0) - (w.x || 0), (m.y || 0) - (w.y || 0));

  let distanceStatus: 'too_far' | 'too_close' | 'optimal' = 'optimal';
  if (palmSpan < 0.12 || boxHeight < 0.20) {
    distanceStatus = 'too_far';
  } else if (palmSpan > 0.40 || boxHeight > 0.70) {
    distanceStatus = 'too_close';
  }

  let framingStatus: 'off_edge' | 'optimal' | 'reposition' = 'optimal';
  if (minX < 0.04 || maxX > 0.96 || minY < 0.04 || maxY > 0.96) {
    framingStatus = 'off_edge';
  } else if (centerX < 0.20 || centerX > 0.80 || centerY < 0.15 || centerY > 0.85) {
    framingStatus = 'reposition';
  }

  const isIdeal = distanceStatus === 'optimal' && framingStatus === 'optimal';

  let guidanceMessage = 'Optimal position — Sign clearly';
  if (framingStatus === 'off_edge') {
    guidanceMessage = 'Hand cut off — Keep hand fully in camera view';
  } else if (distanceStatus === 'too_far') {
    guidanceMessage = 'Move closer to camera for higher accuracy';
  } else if (distanceStatus === 'too_close') {
    guidanceMessage = 'Move slightly back from camera';
  } else if (framingStatus === 'reposition') {
    guidanceMessage = 'Center hand in the guide area';
  }

  return {
    palmSpan,
    boxWidth,
    boxHeight,
    centerX,
    centerY,
    distanceStatus,
    framingStatus,
    isIdeal,
    guidanceMessage,
  };
}
