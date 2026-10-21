/**
 * SETU — RealSign 26-Class Hand Landmark Preprocessor (JavaScript)
 *
 * Normalizes 21 3D hand landmarks into a 63-D rotation- and scale-invariant vector
 * matching SETU's RealSign 26-class feature extractor logic.
 */

/**
 * Normalizes 21 3D hand landmarks into a 63-D rotation- and scale-invariant vector.
 *
 * @param {Array<{x: number, y: number, z?: number}>} landmarks - Array of 21 hand landmarks ({x, y, z})
 * @param {boolean} [isLeftHand=false] - Whether hand is left hand (to mirror X coordinates for canonical geometry)
 * @returns {number[]} 63-element normalized float array
 */
export function normalizeHandLandmarks63D(landmarks, isLeftHand = false) {
  if (!landmarks || landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  // Copy raw points with optional left-hand mirroring
  const pts = landmarks.map((lm) => ({
    x: isLeftHand ? 1.0 - (lm.x || 0) : (lm.x || 0),
    y: lm.y || 0,
    z: lm.z || 0,
  }));

  // 1. Center on Wrist (landmark 0)
  const wrist = pts[0];
  const translated = pts.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  // 2. Scale normalize using Euclidean distance from Wrist (0) to Middle MCP (9)
  const vWristMiddle = translated[9];
  let scale = Math.hypot(vWristMiddle.x, vWristMiddle.y, vWristMiddle.z);
  if (scale < 1e-6) scale = 1.0;

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
  if (normXY > 1e-6) {
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

  if (normYZ > 1e-6 && Math.abs(z9) > 1e-6) {
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
