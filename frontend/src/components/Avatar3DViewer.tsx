import React from 'react';
import { HumanoidAvatar3D } from './HumanoidAvatar3D';

export interface BoneJoint {
  x: number;
  y: number;
  z: number;
}

export interface SkeletonPose3D {
  head: BoneJoint;
  neck: BoneJoint;
  spine: BoneJoint;
  leftShoulder: BoneJoint;
  leftElbow: BoneJoint;
  leftWrist: BoneJoint;
  rightShoulder: BoneJoint;
  rightElbow: BoneJoint;
  rightWrist: BoneJoint;
  rightFingers: BoneJoint[][];
  leftFingers?: BoneJoint[][];
}

export interface Avatar3DViewerProps {
  currentSign: string;
  isTwoHanded?: boolean;
}

/**
 * High-performance 3D Humanoid Avatar Viewer with Full Facial Features,
 * Complete Upper Torso, and 10 Articulated 3D Fingers (Three.js WebGL).
 */
export const Avatar3DViewer: React.FC<Avatar3DViewerProps> = ({ currentSign, isTwoHanded }) => {
  return <HumanoidAvatar3D currentSign={currentSign} isTwoHanded={isTwoHanded} />;
};
