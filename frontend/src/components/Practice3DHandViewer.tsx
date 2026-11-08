import React, { useRef, useEffect, useState, useCallback } from 'react';
import { RotateCw, ZoomIn, ZoomOut, Eye, Layers } from 'lucide-react';

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface HandSkeleton3D {
  wrist: Point3D;
  thumb: [Point3D, Point3D, Point3D, Point3D];   // CMC, MCP, IP, Tip
  index: [Point3D, Point3D, Point3D, Point3D];   // MCP, PIP, DIP, Tip
  middle: [Point3D, Point3D, Point3D, Point3D];  // MCP, PIP, DIP, Tip
  ring: [Point3D, Point3D, Point3D, Point3D];    // MCP, PIP, DIP, Tip
  pinky: [Point3D, Point3D, Point3D, Point3D];   // MCP, PIP, DIP, Tip
}

export interface DualHandSkeleton3D {
  isTwoHanded: boolean;
  rightHand: HandSkeleton3D;
  leftHand?: HandSkeleton3D;
  contactPoints?: Point3D[];
  instructionBadge?: string;
}

interface Practice3DHandViewerProps {
  sign: string;
  className?: string;
}

// Generate anatomically accurate 3D MediaPipe hand joint coordinates for ISL signs
export function generateHand3DCoordinates(rawSign: string): DualHandSkeleton3D {
  const sign = rawSign.toUpperCase().replace(/\s+/g, '_');

  const createHand = (
    wristOffset: Point3D,
    fingerExtensions: [number, number, number, number, number], // 0: fully closed/fist, 1: fully open/extended
    curlAngles: [number, number, number, number, number] = [0, 0, 0, 0, 0], // In-plane curl
    spreadAngles: [number, number, number, number, number] = [-0.25, -0.1, 0, 0.1, 0.22],
    isLeft: boolean = false
  ): HandSkeleton3D => {
    const dir = isLeft ? -1 : 1;
    const wx = wristOffset.x;
    const wy = wristOffset.y;
    const wz = wristOffset.z;

    const baseMcpPositions = [
      { x: wx + dir * 0.18, y: wy + 0.14, z: wz + 0.04 },  // Thumb CMC
      { x: wx + dir * 0.14, y: wy + 0.32, z: wz + 0.02 },  // Index MCP
      { x: wx + dir * 0.04, y: wy + 0.34, z: wz + 0.01 },  // Middle MCP
      { x: wx - dir * 0.06, y: wy + 0.31, z: wz - 0.01 },  // Ring MCP
      { x: wx - dir * 0.15, y: wy + 0.26, z: wz - 0.02 },  // Pinky MCP
    ];

    const generateFinger = (
      _fIdx: number,
      mcpBase: Point3D,
      ext: number,
      curl: number,
      spread: number,
      isThumb: boolean = false
    ): [Point3D, Point3D, Point3D, Point3D] => {
      const joints: Point3D[] = [];
      const segmentLen = isThumb ? 0.09 : 0.085;
      let curr = { ...mcpBase };

      for (let j = 0; j < 4; j++) {
        joints.push({ ...curr });
        // Extension affects curvature: ext=1 -> straight along +Y, ext=0 -> curled inward into palm (+Z)
        const effExt = isThumb && j === 0 ? 1 : ext;
        const effCurl = isThumb ? 0.3 * (1 - ext) : curl;

        const dy = segmentLen * (0.3 + 0.7 * effExt);
        const dz = isThumb ? 0.04 * (1 - effExt) : segmentLen * 0.9 * (1 - effExt) * Math.sin((j + 1) * 0.7);
        const dx = dir * segmentLen * Math.sin(spread) * effExt + dir * (isThumb ? 0.08 * (1 - effExt) : 0);

        curr = {
          x: curr.x + dx,
          y: curr.y + dy - (effCurl * 0.05),
          z: curr.z + dz,
        };
      }
      return [joints[0], joints[1], joints[2], joints[3]];
    };

    return {
      wrist: { x: wx, y: wy, z: wz },
      thumb: generateFinger(0, baseMcpPositions[0], fingerExtensions[0], curlAngles[0], spreadAngles[0], true),
      index: generateFinger(1, baseMcpPositions[1], fingerExtensions[1], curlAngles[1], spreadAngles[1]),
      middle: generateFinger(2, baseMcpPositions[2], fingerExtensions[2], curlAngles[2], spreadAngles[2]),
      ring: generateFinger(3, baseMcpPositions[3], fingerExtensions[3], curlAngles[3], spreadAngles[3]),
      pinky: generateFinger(4, baseMcpPositions[4], fingerExtensions[4], curlAngles[4], spreadAngles[4]),
    };
  };

  // Sign configurations
  let isTwoHanded = false;
  let instructionBadge: string | undefined = undefined;
  let rightHand: HandSkeleton3D;
  let leftHand: HandSkeleton3D | undefined = undefined;
  const contactPoints: Point3D[] = [];

  switch (sign) {
    case 'A':
      // Closed fist, thumb upright alongside index
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [1.0, 0.0, 0.0, 0.0, 0.0]);
      break;

    case 'B':
      // ISL Bimanual: Left palm upright facing right, Right palm touching left
      isTwoHanded = true;
      instructionBadge = 'Two-Handed Contact: Right hand touches Left palm';
      leftHand = createHand({ x: -0.22, y: -0.15, z: 0 }, [0.3, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.16, y: -0.15, z: 0.02 }, [0.3, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]);
      contactPoints.push({ x: -0.03, y: 0.12, z: 0.01 });
      break;

    case 'C':
      // Distinct C-arc curve
      rightHand = createHand(
        { x: 0, y: -0.2, z: 0 },
        [0.55, 0.55, 0.55, 0.55, 0.55],
        [0.2, 0.3, 0.3, 0.3, 0.3],
        [-0.1, -0.05, 0, 0.05, 0.1]
      );
      break;

    case 'D':
      // ISL Bimanual D: Left index points straight up (stem); right index & thumb form the loop
      isTwoHanded = true;
      instructionBadge = 'Two-Handed: Left index stem + Right thumb/index loop';
      leftHand = createHand({ x: -0.2, y: -0.15, z: 0 }, [0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.14, y: -0.15, z: 0 }, [0.7, 0.7, 0, 0, 0], [0.3, 0.3, 0, 0, 0], [-0.1, -0.05, 0, 0, 0]);
      contactPoints.push({ x: -0.05, y: 0.22, z: 0 }, { x: -0.05, y: 0.02, z: 0 });
      break;

    case 'E':
      // Clawed fingers resting on thumb tip
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.25, 0.25, 0.25, 0.25, 0.25]);
      break;

    case 'F':
      // OK-sign circle between thumb & index, 3 fingers upright
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.35, 0.35, 1.0, 1.0, 1.0], [0.2, 0.2, 0, 0, 0]);
      break;

    case 'G':
      // Horizontal index and thumb
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.9, 0.9, 0, 0, 0], [0, 0, 0, 0, 0], [-0.4, -0.4, 0, 0, 0]);
      break;

    case 'H':
      // Horizontal index & middle together
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.2, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0], [-0.3, -0.3, -0.3, 0, 0]);
      break;

    case 'I':
      // Closed fist, Pinky straight up
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 0, 0, 0, 1.0]);
      break;

    case 'J':
      // Pinky extended and curved
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 0, 0, 0, 0.85]);
      break;

    case 'K':
      // Index upright, middle angled forward
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.8, 1.0, 0.8, 0, 0]);
      break;

    case 'L':
      // Distinct L-shape: Thumb at 90 deg, index vertical
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [1.0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0], [-0.75, 0, 0, 0, 0]);
      break;

    case 'M':
      // 3 fingers folded over thumb
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.1, 0.2, 0.2, 0.2, 0]);
      break;

    case 'N':
      // 2 fingers folded over thumb
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.1, 0.2, 0.2, 0, 0]);
      break;

    case 'O':
      // All fingertips touching thumb forming O circle
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.5, 0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3, 0.3]);
      break;

    case 'P':
      // ISL Bimanual P: Left index vertical, right thumb & index loop on top
      isTwoHanded = true;
      instructionBadge = 'Two-Handed: Left index upright + Right upper loop';
      leftHand = createHand({ x: -0.2, y: -0.15, z: 0 }, [0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.12, y: -0.05, z: 0 }, [0.65, 0.65, 0, 0, 0], [0.3, 0.3, 0, 0, 0], [-0.1, -0.05, 0, 0, 0]);
      contactPoints.push({ x: -0.05, y: 0.22, z: 0 });
      break;

    case 'Q':
      // ISL Bimanual Q: Left hand forms circle, Right index points down through it
      isTwoHanded = true;
      instructionBadge = 'Two-Handed: Left O-shape + Right index downwards';
      leftHand = createHand({ x: -0.18, y: -0.08, z: 0 }, [0.5, 0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.18, y: -0.25, z: 0 }, [0, 1.0, 0, 0, 0], [0, 3.14, 0, 0, 0], [0, 0, 0, 0, 0]);
      contactPoints.push({ x: -0.02, y: -0.02, z: 0 });
      break;

    case 'R':
      // Crossed index & middle
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0], [0, 0.05, -0.05, 0, 0]);
      break;

    case 'S':
      // Tight fist, thumb across front
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0]);
      break;

    case 'T':
      // ISL Bimanual T: Left index upright + Right index horizontal crossbar
      isTwoHanded = true;
      instructionBadge = 'Two-Handed Cross: Right index horizontal on Left index';
      leftHand = createHand({ x: -0.18, y: -0.15, z: 0 }, [0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.15, y: 0.05, z: 0.02 }, [0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1.5, 0, 0, 0]);
      contactPoints.push({ x: -0.04, y: 0.20, z: 0.01 });
      break;

    case 'U':
      // Index & middle upright together
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0], [0, -0.02, 0.02, 0, 0]);
      break;

    case 'V':
      // Peace sign / V spread
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0], [0, -0.25, 0.25, 0, 0]);
      break;

    case 'W':
      // 3 fingers upright spread (W)
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 1.0, 1.0, 1.0, 0], [0, 0, 0, 0, 0], [0, -0.25, 0, 0.25, 0]);
      break;

    case 'X':
      // Index hooked / bent
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0, 0.5, 0, 0, 0], [0, 0.8, 0, 0, 0]);
      break;

    case 'Y':
      // Shaka: Thumb and Pinky extended
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [1.0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0], [-0.6, 0, 0, 0, 0.6]);
      break;

    case 'NAMASTE':
      // Both flat hands pressed together at chest
      isTwoHanded = true;
      instructionBadge = 'Two-Handed: Join palms together in prayer position';
      leftHand = createHand({ x: -0.15, y: -0.15, z: 0 }, [0.3, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], true);
      rightHand = createHand({ x: 0.15, y: -0.15, z: 0 }, [0.3, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]);
      contactPoints.push({ x: 0, y: 0.05, z: 0 }, { x: 0, y: 0.22, z: 0 });
      break;

    case 'HELP':
      // Closed fist resting on flat palm
      isTwoHanded = true;
      instructionBadge = 'Two-Handed: Closed right fist resting on flat left palm';
      leftHand = createHand({ x: -0.18, y: -0.2, z: 0 }, [0.4, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, -0.8, -0.8, -0.8, -0.8], true);
      rightHand = createHand({ x: 0.05, y: -0.05, z: 0.05 }, [1.0, 0, 0, 0, 0]);
      contactPoints.push({ x: -0.05, y: -0.05, z: 0.02 });
      break;

    case 'THANK_YOU':
    case 'THANK YOU':
      // Flat hand forward from chin
      rightHand = createHand({ x: 0, y: -0.1, z: 0.1 }, [0.4, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0], [0, -0.1, 0, 0.1, 0.2]);
      break;

    case 'WATER':
      // W-hand tilted toward mouth
      rightHand = createHand({ x: 0.05, y: 0.0, z: 0.1 }, [0, 1.0, 1.0, 1.0, 0], [0, 0, 0, 0, 0], [0, -0.2, 0, 0.2, 0]);
      break;

    case 'FOOD':
      // Pinched fingertips touching lips
      rightHand = createHand({ x: 0, y: -0.05, z: 0.1 }, [0.4, 0.4, 0.4, 0.4, 0.4], [0.3, 0.3, 0.3, 0.3, 0.3]);
      break;

    default:
      // Default open hand
      rightHand = createHand({ x: 0, y: -0.2, z: 0 }, [0.5, 1.0, 1.0, 1.0, 1.0]);
      break;
  }

  return { isTwoHanded, rightHand, leftHand, contactPoints, instructionBadge };
}

export const Practice3DHandViewer: React.FC<Practice3DHandViewerProps> = ({
  sign,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Orbit camera rotation & zoom state
  const [rotX, setRotX] = useState<number>(15); // Pitch (degrees)
  const [rotY, setRotY] = useState<number>(-20); // Yaw (degrees)
  const [zoom, setZoom] = useState<number>(1.05);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const lastMousePos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Reset to ideal 3/4 angle
  const handleResetCamera = useCallback(() => {
    setRotX(15);
    setRotY(-20);
    setZoom(1.05);
  }, []);

  // Mouse drag orbit controls
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMousePos.current.x;
    const dy = e.clientY - lastMousePos.current.y;
    setRotY((prev) => prev + dx * 0.7);
    setRotX((prev) => Math.max(-60, Math.min(60, prev + dy * 0.7)));
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => setIsDragging(false);

  // Wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((prev) => Math.max(0.7, Math.min(1.8, prev - e.deltaY * 0.001)));
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const skeleton = generateHand3DCoordinates(sign);

    // 3D rotation matrix calculation
    const radX = (rotX * Math.PI) / 180;
    const radY = (rotY * Math.PI) / 180;
    const cosX = Math.cos(radX);
    const sinX = Math.sin(radX);
    const cosY = Math.cos(radY);
    const sinY = Math.sin(radY);

    const transform3D = (pt: Point3D) => {
      // 1. Rotate Y (Yaw)
      const x1 = pt.x * cosY + pt.z * sinY;
      const y1 = pt.y;
      const z1 = -pt.x * sinY + pt.z * cosY;

      // 2. Rotate X (Pitch)
      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      // 3. Perspective Projection
      const focalLength = 460 * zoom;
      const cameraDistance = 1.35;
      const zEff = z2 + cameraDistance;
      const scale = focalLength / Math.max(0.2, zEff);

      return {
        x: width / 2 + x2 * scale,
        y: height / 2 - y2 * scale,
        z: z2,
        scale,
      };
    };

    // Render single hand with joints and bones
    const renderHand = (hand: HandSkeleton3D, primaryColor: string, secondaryColor: string) => {
      const fingers = [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky];

      // Step 1: Draw Palm Metacarpal Webbing Fill (Glass translucent mesh)
      const wProj = transform3D(hand.wrist);
      const tMcpProj = transform3D(hand.thumb[0]);
      const iMcpProj = transform3D(hand.index[0]);
      const pMcpProj = transform3D(hand.pinky[0]);

      ctx.beginPath();
      ctx.moveTo(wProj.x, wProj.y);
      ctx.lineTo(tMcpProj.x, tMcpProj.y);
      ctx.lineTo(iMcpProj.x, iMcpProj.y);
      ctx.lineTo(pMcpProj.x, pMcpProj.y);
      ctx.closePath();
      ctx.fillStyle = primaryColor === '#06B6D4' ? 'rgba(6, 182, 212, 0.08)' : 'rgba(129, 140, 248, 0.08)';
      ctx.fill();

      // Step 2: Draw Palm Carpal Bones (Wrist to MCP joints)
      [hand.thumb[0], hand.index[0], hand.middle[0], hand.ring[0], hand.pinky[0]].forEach((mcp) => {
        const pMcp = transform3D(mcp);
        ctx.beginPath();
        ctx.moveTo(wProj.x, wProj.y);
        ctx.lineTo(pMcp.x, pMcp.y);
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        ctx.lineWidth = 2.5 * (wProj.scale / 400);
        ctx.stroke();
      });

      // Step 3: Draw Phalange Bones
      fingers.forEach((fingerChain) => {
        for (let i = 0; i < fingerChain.length - 1; i++) {
          const p1 = transform3D(fingerChain[i]);
          const p2 = transform3D(fingerChain[i + 1]);

          const grad = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
          grad.addColorStop(0, primaryColor);
          grad.addColorStop(1, secondaryColor);

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = grad;
          ctx.lineWidth = Math.max(2, 4.5 * (p1.scale / 400));
          ctx.lineCap = 'round';
          ctx.stroke();
        }
      });

      // Step 4: Draw Joint Spheres (Depth-sorted)
      const allJoints: { pt: Point3D; isTip: boolean }[] = [{ pt: hand.wrist, isTip: false }];
      fingers.forEach((chain) => {
        chain.forEach((joint, idx) => {
          allJoints.push({ pt: joint, isTip: idx === chain.length - 1 });
        });
      });

      allJoints.sort((a, b) => a.pt.z - b.pt.z);

      allJoints.forEach(({ pt, isTip }) => {
        const proj = transform3D(pt);
        const radius = isTip ? Math.max(3.5, 6 * (proj.scale / 400)) : Math.max(2.5, 4.5 * (proj.scale / 400));

        // Glow ring
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, radius * 1.6, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? 'rgba(34, 211, 238, 0.25)' : 'rgba(148, 163, 184, 0.15)';
        ctx.fill();

        // Solid joint sphere
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? '#22D3EE' : '#F8FAFC';
        ctx.fill();
        ctx.strokeStyle = '#0F172A';
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });
    };

    // Render Left Hand if dual-handed
    if (skeleton.isTwoHanded && skeleton.leftHand) {
      renderHand(skeleton.leftHand, '#818CF8', '#C084FC');
    }

    // Render Dominant Right Hand
    renderHand(skeleton.rightHand, '#06B6D4', '#10B981');

    // Draw Golden Contact Rings for two-handed touch points
    if (skeleton.contactPoints && skeleton.contactPoints.length > 0) {
      skeleton.contactPoints.forEach((cp) => {
        const cpProj = transform3D(cp);
        ctx.beginPath();
        ctx.arc(cpProj.x, cpProj.y, 14 * (cpProj.scale / 400), 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(cpProj.x, cpProj.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#F59E0B';
        ctx.fill();
      });
    }
  }, [sign, rotX, rotY, zoom]);

  const skeletonInfo = generateHand3DCoordinates(sign);

  return (
    <div className={`practice-3d-visualizer-container ${className}`}>
      {/* 3D Hand Canvas */}
      <div
        className="canvas-interactive-wrapper"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        title="Click & drag to rotate 3D hand skeleton in 360°"
      >
        <canvas
          ref={canvasRef}
          width={360}
          height={260}
          className="practice-3d-canvas"
        />

        {/* 3D Orbit Indicator Overlay */}
        <div className="canvas-badge-overlay">
          <span className="hand-model-badge">
            <Layers size={11} className="text-cyan" />
            <span>3D ISL Canonical Rig</span>
          </span>
          <span className="orbit-hint">Drag to rotate 360°</span>
        </div>

        {/* View Controls Toolbar */}
        <div className="canvas-view-controls">
          <button
            type="button"
            className="view-tool-btn"
            onClick={() => setZoom((z) => Math.min(1.8, z + 0.15))}
            title="Zoom In"
            aria-label="Zoom In"
          >
            <ZoomIn size={13} />
          </button>
          <button
            type="button"
            className="view-tool-btn"
            onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}
            title="Zoom Out"
            aria-label="Zoom Out"
          >
            <ZoomOut size={13} />
          </button>
          <button
            type="button"
            className="view-tool-btn"
            onClick={handleResetCamera}
            title="Reset to 3/4 Perspective"
            aria-label="Reset Camera"
          >
            <RotateCw size={13} />
          </button>
        </div>
      </div>

      {/* Dual Hand / Contact Guidance Callout */}
      {skeletonInfo.instructionBadge && (
        <div className="practice-contact-callout">
          <Eye size={12} className="text-amber" />
          <span>{skeletonInfo.instructionBadge}</span>
        </div>
      )}
    </div>
  );
};

export default Practice3DHandViewer;
