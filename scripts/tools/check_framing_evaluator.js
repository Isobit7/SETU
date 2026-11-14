/**
 * SETU — Test Check for Client-Side Framing Evaluator
 * Validates distance constraints, centering heuristics, and hand-presence gating.
 */
import assert from 'node:assert';
import { evaluateFraming } from '../frontend/src/utils/framingEvaluator.ts';

console.log('Testing SETU Framing Evaluator...');

// Helper to create 21 dummy hand points
function createHand(wristX, wristY, tipY) {
  const pts = [];
  for (let i = 0; i < 21; i++) {
    pts.push({ x: wristX, y: wristY, z: 0 });
  }
  pts[12] = { x: wristX, y: tipY, z: 0 }; // middle fingertip
  return pts;
}

// 1. Null / Empty
const emptyRes = evaluateFraming(null);
assert.strictEqual(emptyRes.status, 'searching');
assert.strictEqual(emptyRes.isOptimal, false);
console.log('  ✓ Null / Empty landmark handling passed');

// 2. Optimal Pose & Hands
const optimalPose = [];
for (let i = 0; i < 33; i++) optimalPose.push({ x: 0.5, y: 0.5, z: 0 });
optimalPose[0] = { x: 0.5, y: 0.20, z: 0 }; // nose
optimalPose[11] = { x: 0.35, y: 0.38, z: 0 }; // left shoulder
optimalPose[12] = { x: 0.65, y: 0.38, z: 0 }; // right shoulder (span = 0.30)

const optimalRes = evaluateFraming({
  pose: optimalPose,
  left_hand: createHand(0.40, 0.60, 0.45),
  right_hand: createHand(0.60, 0.60, 0.45),
  face: [],
});
assert.strictEqual(optimalRes.status, 'optimal');
assert.strictEqual(optimalRes.isOptimal, true);
assert.ok(optimalRes.score >= 85, 'Score should be high for optimal framing');
console.log('  ✓ Optimal pose & hands framing passed (Score:', optimalRes.score, ')');

// 3. User Too Far (Shoulder span too narrow)
const farPose = optimalPose.map(p => ({ ...p }));
farPose[11] = { x: 0.45, y: 0.38, z: 0 };
farPose[12] = { x: 0.55, y: 0.38, z: 0 }; // span = 0.10 (< 0.17)
const farRes = evaluateFraming({
  pose: farPose,
  left_hand: createHand(0.45, 0.60, 0.52),
  right_hand: [],
  face: [],
});
assert.strictEqual(farRes.status, 'step_closer');
assert.strictEqual(farRes.isOptimal, false);
console.log('  ✓ Step closer constraint passed');

// 4. User Too Close (Shoulder span too wide)
const closePose = optimalPose.map(p => ({ ...p }));
closePose[11] = { x: 0.15, y: 0.38, z: 0 };
closePose[12] = { x: 0.85, y: 0.38, z: 0 }; // span = 0.70 (> 0.56)
const closeRes = evaluateFraming({
  pose: closePose,
  left_hand: createHand(0.35, 0.60, 0.45),
  right_hand: [],
  face: [],
});
assert.strictEqual(closeRes.status, 'step_back');
assert.strictEqual(closeRes.isOptimal, false);
console.log('  ✓ Move back constraint passed');

// 5. User Off-Center (Shifted left/right)
const offCenterPose = optimalPose.map(p => ({ ...p }));
offCenterPose[11] = { x: 0.10, y: 0.38, z: 0 };
offCenterPose[12] = { x: 0.38, y: 0.38, z: 0 }; // center ~ 0.24 (< 0.33)
const offCenterRes = evaluateFraming({
  pose: offCenterPose,
  left_hand: createHand(0.20, 0.60, 0.45),
  right_hand: [],
  face: [],
}, { mirrored: true });
assert.strictEqual(offCenterRes.status, 'center_yourself');
assert.ok(offCenterRes.message.includes('Center Yourself'));
console.log('  ✓ Center yourself constraint passed');

// 6. Pose without hands detected
const noHandsRes = evaluateFraming({
  pose: optimalPose,
  left_hand: [],
  right_hand: [],
  face: [],
});
assert.strictEqual(noHandsRes.status, 'raise_hands');
assert.strictEqual(noHandsRes.isOptimal, false);
console.log('  ✓ Raise hands gating check passed');

// 7. Hands too low (below subtitle threshold)
const lowHandsRes = evaluateFraming({
  pose: optimalPose,
  left_hand: createHand(0.50, 0.95, 0.90), // wrist at y=0.95
  right_hand: [],
  face: [],
});
assert.strictEqual(lowHandsRes.status, 'hands_too_low');
assert.strictEqual(lowHandsRes.isOptimal, false);
console.log('  ✓ Hands too low check passed');

// 8. Hand-only fallback (no pose)
const handOnlyRes = evaluateFraming({
  pose: [],
  left_hand: createHand(0.50, 0.55, 0.35),
  right_hand: [],
  face: [],
});
assert.strictEqual(handOnlyRes.status, 'optimal');
assert.strictEqual(handOnlyRes.isOptimal, true);
console.log('  ✓ Hand-only fallback passed');

console.log('All 8 framing evaluator checks passed successfully! ✨');
