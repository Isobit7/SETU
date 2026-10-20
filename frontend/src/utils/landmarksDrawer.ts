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

  // 1. Draw Left Hand (Emerald / Bright Green #10B981)
  if (styles.showLeftHand && landmarks.left_hand && landmarks.left_hand.length > 0) {
    const pts = landmarks.left_hand;

    // Hand skeleton connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#10B981';
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
      const radius = isFingertip ? 5.5 : 3.5;

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isFingertip ? '#34D399' : '#10B981';
      ctx.fill();

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#064E3B';
      ctx.stroke();
    }
  }

  // 2. Draw Right Hand (Amber / Bright Orange #F59E0B)
  if (styles.showRightHand && landmarks.right_hand && landmarks.right_hand.length > 0) {
    const pts = landmarks.right_hand;

    // Hand skeleton connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#F59E0B';
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
      const radius = isFingertip ? 5.5 : 3.5;

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isFingertip ? '#FDE68A' : '#F59E0B';
      ctx.fill();

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#78350F';
      ctx.stroke();
    }
  }

  // NOTE: Face landmarks, body/pose landmarks, head circles, torso lines, and pose bounding boxes
  // are explicitly excluded from visualization per requirements to ensure a clean, distraction-free
  // hand-landmarks-only overlay.
}
