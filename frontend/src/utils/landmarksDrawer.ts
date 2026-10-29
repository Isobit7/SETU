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
  styles: LandmarkStyles,
  detectedObjects?: Array<{
    label: string;
    display_name: string;
    confidence: number;
    is_held: boolean;
    bbox: [number, number, number, number];
  }>
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

    // Batched skeleton connections in a single path
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (const [s, e] of HAND_CONNECTIONS) {
      if (s < pts.length && e < pts.length) {
        const [x1, y1] = getPt(pts[s]);
        const [x2, y2] = getPt(pts[e]);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
    ctx.stroke();

    // Batched standard joints
    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      if (![4, 8, 12, 16, 20].includes(i)) {
        const [px, py] = getPt(pts[i]);
        ctx.moveTo(px + 2.5, py);
        ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
      }
    }
    ctx.fill();

    // Batched fingertips
    ctx.fillStyle = '#7dd3fc';
    ctx.beginPath();
    for (const i of [4, 8, 12, 16, 20]) {
      if (i < pts.length) {
        const [px, py] = getPt(pts[i]);
        ctx.moveTo(px + 4.5, py);
        ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
      }
    }
    ctx.fill();
  }

  // 2. Draw Right Hand (Vibrant Emerald / Mint Green: #34d399)
  if (styles.showRightHand && landmarks.right_hand && landmarks.right_hand.length > 0) {
    const pts = landmarks.right_hand;

    // Batched skeleton connections in a single path
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#34d399';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (const [s, e] of HAND_CONNECTIONS) {
      if (s < pts.length && e < pts.length) {
        const [x1, y1] = getPt(pts[s]);
        const [x2, y2] = getPt(pts[e]);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
    ctx.stroke();

    // Batched standard joints
    ctx.fillStyle = '#059669';
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      if (![4, 8, 12, 16, 20].includes(i)) {
        const [px, py] = getPt(pts[i]);
        ctx.moveTo(px + 2.5, py);
        ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
      }
    }
    ctx.fill();

    // Batched fingertips
    ctx.fillStyle = '#a7f3d0';
    ctx.beginPath();
    for (const i of [4, 8, 12, 16, 20]) {
      if (i < pts.length) {
        const [px, py] = getPt(pts[i]);
        ctx.moveTo(px + 4.5, py);
        ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
      }
    }
    ctx.fill();
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

  // 4. Detected Physical Objects Bounding Boxes (Show-and-Sign Multimodal Grounding)
  if (detectedObjects && detectedObjects.length > 0) {
    for (const obj of detectedObjects) {
      // Only draw bounding boxes for held objects actively participating in Show-and-Sign
      if (!obj.is_held || !obj.bbox || obj.bbox.length < 4) continue;
      const [ymin, xmin, ymax, xmax] = obj.bbox;

      // Coordinate mapping with mirror horizontal flip support
      const leftNorm = styles.mirrored ? (1.0 - xmax) : xmin;
      const rightNorm = styles.mirrored ? (1.0 - xmin) : xmax;
      const x1 = Math.max(0, leftNorm * width);
      const y1 = Math.max(0, ymin * height);
      const boxW = Math.min(width - x1, Math.max(20, (rightNorm - leftNorm) * width));
      const boxH = Math.min(height - y1, Math.max(20, (ymax - ymin) * height));

      const isHeld = obj.is_held;
      const strokeColor = isHeld ? '#f59e0b' : '#38bdf8'; // Amber Gold if held near hand, else Sky Cyan
      const bgColor = isHeld ? 'rgba(245, 158, 11, 0.12)' : 'rgba(56, 189, 248, 0.08)';

      ctx.save();
      // Bounding box outline
      ctx.lineWidth = isHeld ? 2.5 : 1.8;
      ctx.strokeStyle = strokeColor;
      ctx.fillStyle = bgColor;
      ctx.beginPath();
      ctx.roundRect(x1, y1, boxW, boxH, 8);
      ctx.fill();
      ctx.stroke();

      // Top label badge
      const tagText = `${obj.display_name} ${Math.round(obj.confidence * 100)}%${isHeld ? ' • HELD' : ''}`;
      ctx.font = '600 11px system-ui, -apple-system, sans-serif';
      const textMetrics = ctx.measureText(tagText);
      const badgeW = textMetrics.width + 14;
      const badgeH = 20;
      const badgeY = Math.max(2, y1 - badgeH - 2);

      ctx.fillStyle = strokeColor;
      ctx.beginPath();
      ctx.roundRect(x1, badgeY, badgeW, badgeH, 4);
      ctx.fill();

      ctx.fillStyle = '#0f172a';
      ctx.fillText(tagText, x1 + 7, badgeY + 14);
      ctx.restore();
    }
  }

  // NOTE: Face landmarks, body/pose landmarks, head circles, torso lines, and pose bounding boxes
  // are explicitly excluded from visualization per requirements to ensure a clean, distraction-free
  // hand-landmarks-only overlay.
}
