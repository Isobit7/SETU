import type { LandmarkStyles } from '../types';

// Hand skeleton 21-point topology connections (MediaPipe Hands standard)
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // Index
  [5, 9], [9, 10], [10, 11], [11, 12],   // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                                // Palm base
];

/**
 * Draws ONLY hand landmarks and hand skeleton connections.
 * Completely excludes face, body/pose, shoulders, torso, and non-hand dots.
 */
export function drawLandmarksOnCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  landmarks: {
    left_hand?: Array<{ x: number; y: number; z: number }>;
    right_hand?: Array<{ x: number; y: number; z: number }>;
    pose?: Array<{ x: number; y: number; z: number }>;
    face?: Array<{ x: number; y: number; z: number }>;
  },
  styles: LandmarkStyles
) {
  ctx.clearRect(0, 0, width, height);

  const getPt = (p: { x: number; y: number }): [number, number] => {
    // When video is mirrored, flip X coordinate horizontally to match mirror preview
    const normX = styles.mirrored ? (1.0 - p.x) : p.x;
    return [normX * width, p.y * height];
  };

  // 1. Draw Left Hand (Vibrant Cyan / Sky Blue: #38bdf8)
  if (styles.showLeftHand && landmarks.left_hand && landmarks.left_hand.length > 0) {
    const pts = landmarks.left_hand;

    // Hand skeleton connections
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [s, e] of HAND_CONNECTIONS) {
      if (s < pts.length && e < pts.length) {
        const [x1, y1] = getPt(pts[s]);
        const [x2, y2] = getPt(pts[e]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // Hand joints
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = getPt(pts[i]);
      const isFingertip = [4, 8, 12, 16, 20].includes(i);
      const radius = isFingertip ? 4.5 : 2.5;

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isFingertip ? '#7dd3fc' : '#0284c7';
      ctx.fill();

      ctx.lineWidth = 1.0;
      ctx.strokeStyle = '#082f49';
      ctx.stroke();
    }
  }

  // 2. Draw Right Hand (Vibrant Emerald / Mint Green: #34d399)
  if (styles.showRightHand && landmarks.right_hand && landmarks.right_hand.length > 0) {
    const pts = landmarks.right_hand;

    // Hand skeleton connections
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#34d399';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [s, e] of HAND_CONNECTIONS) {
      if (s < pts.length && e < pts.length) {
        const [x1, y1] = getPt(pts[s]);
        const [x2, y2] = getPt(pts[e]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // Hand joints
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = getPt(pts[i]);
      const isFingertip = [4, 8, 12, 16, 20].includes(i);
      const radius = isFingertip ? 4.5 : 2.5;

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isFingertip ? '#a7f3d0' : '#059669';
      ctx.fill();

      ctx.lineWidth = 1.0;
      ctx.strokeStyle = '#022c22';
      ctx.stroke();
    }
  }

  // 3. Inter-Hand Contact & Proximity Indicator (Crucial for B, D, P, Q, T Two-Handed ISL Gestures)
  if (
    styles.showLeftHand &&
    styles.showRightHand &&
    landmarks.left_hand &&
    landmarks.left_hand.length > 0 &&
    landmarks.right_hand &&
    landmarks.right_hand.length > 0
  ) {
    const leftPts = landmarks.left_hand;
    const rightPts = landmarks.right_hand;
    const CONTACT_DIST_PX = 32.0;
    const keyIndices = [0, 4, 8, 12, 16, 20]; // Wrists and fingertips

    for (const li of keyIndices) {
      if (li >= leftPts.length) continue;
      const [lx, ly] = getPt(leftPts[li]);

      for (const ri of keyIndices) {
        if (ri >= rightPts.length) continue;
        const [rx, ry] = getPt(rightPts[ri]);
        const dist = Math.hypot(lx - rx, ly - ry);

        if (dist <= CONTACT_DIST_PX) {
          const midX = (lx + rx) / 2;
          const midY = (ly + ry) / 2;

          // Glowing contact pulse indicator
          ctx.save();
          ctx.beginPath();
          ctx.arc(midX, midY, 8.0, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(236, 72, 153, 0.45)'; // Vibrant Rose pulse
          ctx.fill();

          ctx.beginPath();
          ctx.arc(midX, midY, 3.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#f43f5e';
          ctx.fill();

          // Delicate connecting bridge
          ctx.beginPath();
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#f43f5e';
          ctx.moveTo(lx, ly);
          ctx.lineTo(rx, ry);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  }

  // NOTE: Face landmarks, body/pose landmarks, head circles, torso lines, and pose bounding boxes
  // are explicitly excluded from visualization per requirements to ensure a clean, distraction-free
  // hand-landmarks-only overlay.
}
