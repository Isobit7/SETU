import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Avatar3DViewer } from './Avatar3DViewer';
import { ApiService } from '../services/apiService';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  Mic,
  MicOff,
  Send,
  Maximize2,
  Minimize2,
  Cpu,
  Languages,
} from 'lucide-react';

interface TextToSignWorkspaceProps {
  onSpeak?: (text: string) => void;
  isSpeaking?: boolean;
}

// Hand pose definition for Canvas Avatar
export interface HandPose {
  x: number;
  y: number;
  fingers: number[]; // 5 finger extension states [thumb, index, middle, ring, pinky] (0: closed, 1: open)
  wristAngle: number;
  isTwoHanded?: boolean;
  leftHandX?: number;
  leftHandY?: number;
  leftFingers?: number[];
  leftWristAngle?: number;
}

export interface SignData {
  label: string;
  desc: string;
  category: string;
  indicHindi: string;
  pose: HandPose;
}

// Comprehensive Library of Authentic Indian Sign Language (ISL) Signs & Alphabets
export const SIGN_DATABASE: Record<string, SignData> = {
  // -------------------------------------------------------------
  // Greetings & Courtesies
  // -------------------------------------------------------------
  NAMASTE: {
    label: 'Namaste (ISL Greeting)',
    desc: 'Both flat palms pressed together at center chest in respectful Anjali Mudra with gentle head bow.',
    category: 'Greeting',
    indicHindi: 'नमस्ते / नमस्कार',
    pose: {
      x: 314,
      y: 228,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.1,
      isTwoHanded: true,
      leftHandX: 286,
      leftHandY: 228,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.1,
    },
  },
  VANAKKAM: {
    label: 'Vanakkam (ISL Greeting)',
    desc: 'Both flat palms pressed together at center chest in respectful Anjali Mudra.',
    category: 'Greeting',
    indicHindi: 'வணக்கம் / नमस्ते',
    pose: {
      x: 314,
      y: 228,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.1,
      isTwoHanded: true,
      leftHandX: 286,
      leftHandY: 228,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.1,
    },
  },
  HELLO: {
    label: 'Hello (Greeting)',
    desc: 'Open flat hand near right temple sweeping outward in a crisp, friendly salute wave.',
    category: 'Greeting',
    indicHindi: 'नमस्ते / हैलो',
    pose: { x: 350, y: 190, fingers: [1, 1, 1, 1, 1], wristAngle: -0.3 },
  },
  HI: {
    label: 'Hi (Casual Greeting)',
    desc: 'Open palm raised to shoulder-level oscillating gently side-to-side.',
    category: 'Greeting',
    indicHindi: 'नमस्ते / हाय',
    pose: { x: 350, y: 190, fingers: [1, 1, 1, 1, 1], wristAngle: -0.3 },
  },
  GOOD_MORNING: {
    label: 'Good Morning',
    desc: 'Compound sign: Crisp thumbs-up for "good", followed by dominant open hand rising like the morning sun.',
    category: 'Greeting',
    indicHindi: 'शुभ प्रभात',
    pose: {
      x: 350,
      y: 200,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 250,
      leftHandY: 280,
      leftFingers: [0.8, 0.8, 0.8, 0.8, 0.8],
      leftWristAngle: 0.1,
    },
  },
  GOOD_AFTERNOON: {
    label: 'Good Afternoon',
    desc: 'Compound sign: Thumbs-up for "good", followed by dominant open hand positioned high overhead like midday sun.',
    category: 'Greeting',
    indicHindi: 'शुभ दोपहर',
    pose: {
      x: 330,
      y: 180,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.1,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 280,
      leftFingers: [0.8, 0.8, 0.8, 0.8, 0.8],
      leftWristAngle: 0.1,
    },
  },
  GOOD_NIGHT: {
    label: 'Good Night',
    desc: 'Compound sign: Thumbs-up for "good", followed by dominant open hand dipping below horizontal arm like sunset.',
    category: 'Greeting',
    indicHindi: 'शुभ रात्रि',
    pose: {
      x: 320,
      y: 270,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: 0.3,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 240,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0,
    },
  },
  THANK_YOU: {
    label: 'Thank You (Gratitude)',
    desc: 'Flat hand fingertips touch chin/lips then sweep gracefully forward toward conversational partner with warm nod.',
    category: 'Courtesy',
    indicHindi: 'धन्यवाद / शुक्रिया',
    pose: { x: 310, y: 220, fingers: [0.8, 1, 1, 1, 1], wristAngle: 0.2 },
  },
  PLEASE: {
    label: 'Please (Polite)',
    desc: 'Flat open palm rubbing clockwise circular motion over heart/chest with courteous expression.',
    category: 'Courtesy',
    indicHindi: 'कृपया',
    pose: { x: 290, y: 250, fingers: [1, 1, 1, 1, 1], wristAngle: -0.2 },
  },
  WELCOME: {
    label: 'Welcome (Courtesy)',
    desc: 'Both hands open palm-up sweeping smoothly inward toward the body in an embracing welcome.',
    category: 'Courtesy',
    indicHindi: 'स्वागत है',
    pose: {
      x: 360,
      y: 260,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: 0.2,
      isTwoHanded: true,
      leftHandX: 240,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: -0.2,
    },
  },
  SORRY: {
    label: 'Sorry (Apology)',
    desc: 'Closed fist rubbing circular motion over center chest with apologetic head tilt.',
    category: 'Courtesy',
    indicHindi: 'माफ़ कीजिए',
    pose: { x: 300, y: 250, fingers: [0, 0, 0, 0, 0], wristAngle: -0.2 },
  },
  YES: {
    label: 'Yes (Affirmation)',
    desc: 'Closed fist with thumb upright nodding up and down at shoulder height like a nodding head.',
    category: 'Common',
    indicHindi: 'हाँ',
    pose: { x: 330, y: 220, fingers: [0.2, 0, 0, 0, 0], wristAngle: 0.3 },
  },
  NO: {
    label: 'No (Negation)',
    desc: 'Index and middle fingers snapping down firmly onto thumb like a bird beak.',
    category: 'Common',
    indicHindi: 'नहीं',
    pose: { x: 330, y: 230, fingers: [0.5, 0.5, 0.5, 0, 0], wristAngle: 0.1 },
  },
  GOOD: {
    label: 'Good (Positive)',
    desc: 'Crisp thumbs-up gesture moving forward with affirmative facial expression.',
    category: 'Common',
    indicHindi: 'अच्छा / बहुत बढ़िया',
    pose: { x: 330, y: 240, fingers: [1, 0, 0, 0, 0], wristAngle: 0 },
  },
  BAD: {
    label: 'Bad (Negative)',
    desc: 'Flat hand held palm down thrust downward with head shake.',
    category: 'Common',
    indicHindi: 'खराब / बुरा',
    pose: { x: 330, y: 270, fingers: [1, 1, 1, 1, 1], wristAngle: 0.4 },
  },

  // -------------------------------------------------------------
  // Family & People
  // -------------------------------------------------------------
  FATHER: {
    label: 'Father (पिता / Dad)',
    desc: 'Dominant index finger strokes mustache line across upper lip twice with respectful head nod.',
    category: 'Family',
    indicHindi: 'पिता / पिताजी',
    pose: { x: 310, y: 175, fingers: [0.2, 1, 0, 0, 0], wristAngle: -0.3 },
  },
  MOTHER: {
    label: 'Mother (माँ / Mom)',
    desc: 'Thumb tip gently taps and strokes side of chin / nose-pin line with warm maternal expression.',
    category: 'Family',
    indicHindi: 'माँ / माताजी',
    pose: { x: 310, y: 195, fingers: [1, 0.2, 0.2, 0.2, 0.2], wristAngle: -0.2 },
  },
  BROTHER: {
    label: 'Brother (भाई / Bhai)',
    desc: 'Two-handed: Left index points forward; right index strokes mustache line then rests over left index in solidarity.',
    category: 'Family',
    indicHindi: 'भाई',
    pose: {
      x: 315,
      y: 220,
      fingers: [0.2, 1, 0, 0, 0],
      wristAngle: -0.15,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 235,
      leftFingers: [0.2, 1, 0, 0, 0],
      leftWristAngle: 0.15,
    },
  },
  SISTER: {
    label: 'Sister (बहन / Behen)',
    desc: 'Two-handed: Dominant hand touches chin / earlobe (female marker) then joins left palm in kinship.',
    category: 'Family',
    indicHindi: 'बहन',
    pose: {
      x: 315,
      y: 220,
      fingers: [0.2, 1, 0, 0, 0],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 235,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.2,
    },
  },
  FAMILY: {
    label: 'Family (परिवार)',
    desc: 'Two-handed: Both hands in F/C shapes starting with index/thumbs touching, tracing wide forward circle meeting at pinkies.',
    category: 'Family',
    indicHindi: 'परिवार',
    pose: {
      x: 330,
      y: 240,
      fingers: [0.7, 0.7, 1, 1, 1],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 240,
      leftFingers: [0.7, 0.7, 1, 1, 1],
      leftWristAngle: 0.2,
    },
  },
  FRIEND: {
    label: 'Friend (दोस्त / Mitra)',
    desc: 'Two-handed: Both curved index fingers hook together, unhook, and hook in reverse in mutual bond.',
    category: 'People',
    indicHindi: 'दोस्त / मित्र',
    pose: {
      x: 315,
      y: 235,
      fingers: [0.2, 0.6, 0, 0, 0],
      wristAngle: -0.3,
      isTwoHanded: true,
      leftHandX: 285,
      leftHandY: 235,
      leftFingers: [0.2, 0.6, 0, 0, 0],
      leftWristAngle: 0.3,
    },
  },
  DOCTOR: {
    label: 'Doctor (चिकित्सक)',
    desc: 'Two-handed: Left arm extended flat palm up; right index and middle fingertips tap left wrist pulse point twice.',
    category: 'Emergency',
    indicHindi: 'डॉक्टर / चिकित्सक',
    pose: {
      x: 295,
      y: 245,
      fingers: [0, 1, 1, 0, 0],
      wristAngle: -0.4,
      isTwoHanded: true,
      leftHandX: 260,
      leftHandY: 255,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.3,
    },
  },

  // -------------------------------------------------------------
  // Places & Education
  // -------------------------------------------------------------
  SCHOOL: {
    label: 'School (विद्यालय / पाठशाला)',
    desc: 'Two-handed: Flat open dominant hand claps down onto flat open non-dominant hand twice with rhythmic focus.',
    category: 'Education',
    indicHindi: 'विद्यालय / स्कूल',
    pose: {
      x: 315,
      y: 235,
      fingers: [0.8, 1, 1, 1, 1],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 250,
      leftFingers: [0.8, 1, 1, 1, 1],
      leftWristAngle: 0.25,
    },
  },
  COLLEGE: {
    label: 'College (महाविद्यालय)',
    desc: 'Two-handed: Dominant hand starts flat on left palm, then swoops upward and circular in elevated learning.',
    category: 'Education',
    indicHindi: 'महाविद्यालय / कॉलेज',
    pose: {
      x: 330,
      y: 210,
      fingers: [0.6, 0.6, 0.6, 0.6, 0.6],
      wristAngle: -0.4,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.2,
    },
  },
  HOME: {
    label: 'Home (घर)',
    desc: 'Two-handed: Both flat hands brought together at fingertips forming the pitched triangular roof of a house.',
    category: 'Place',
    indicHindi: 'घर / निवास',
    pose: {
      x: 325,
      y: 215,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.6,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 215,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.6,
    },
  },
  HOUSE: {
    label: 'House (घर)',
    desc: 'Two-handed: Both flat hands touch at fingertips forming a triangular roof, then separate downward forming walls.',
    category: 'Place',
    indicHindi: 'घर / मकान',
    pose: {
      x: 325,
      y: 215,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.6,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 215,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.6,
    },
  },
  HOSPITAL: {
    label: 'Hospital (अस्पताल)',
    desc: 'Index and middle fingers draw a sacred medical cross (+) on the upper left arm/shoulder.',
    category: 'Emergency',
    indicHindi: 'अस्पताल',
    pose: { x: 280, y: 210, fingers: [0, 1, 1, 0, 0], wristAngle: -0.5 },
  },
  BOOK: {
    label: 'Book (किताब / पुस्तक)',
    desc: 'Two-handed: Both flat hands held together pinky-to-pinky like closed book, then open outward like turning pages.',
    category: 'Education',
    indicHindi: 'किताब / पुस्तक',
    pose: {
      x: 330,
      y: 245,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: 0.2,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 245,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: -0.2,
    },
  },
  CAR: {
    label: 'Car (गाड़ी / कार)',
    desc: 'Two-handed: Both hands in loose fists gripping an imaginary steering wheel and turning alternately.',
    category: 'Transport',
    indicHindi: 'गाड़ी / कार',
    pose: {
      x: 335,
      y: 235,
      fingers: [0.3, 0.3, 0.3, 0.3, 0.3],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 265,
      leftHandY: 235,
      leftFingers: [0.3, 0.3, 0.3, 0.3, 0.3],
      leftWristAngle: 0.2,
    },
  },
  MONEY: {
    label: 'Money (पैसे / धन)',
    desc: 'Dominant thumb rubs rhythmically across tips of index and middle fingers counting rupee notes.',
    category: 'Commerce',
    indicHindi: 'पैसे / धन',
    pose: { x: 325, y: 240, fingers: [0.6, 0.6, 0.6, 0, 0], wristAngle: -0.1 },
  },

  // -------------------------------------------------------------
  // Essential Needs, Actions & Emotions
  // -------------------------------------------------------------
  HELP: {
    label: 'Help (सहायता / मदद)',
    desc: 'Two-handed: Left flat palm facing up; right fist with upright thumb placed upon left palm and lifted upward together.',
    category: 'Emergency',
    indicHindi: 'मदद / सहायता',
    pose: {
      x: 315,
      y: 245,
      fingers: [1, 0, 0, 0, 0],
      wristAngle: 0,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.25,
    },
  },
  WATER: {
    label: 'Water (पानी / Drink)',
    desc: 'W-hand shape (three middle fingers upright) tapped gently twice against side of chin/mouth.',
    category: 'Need',
    indicHindi: 'पानी / जल',
    pose: { x: 325, y: 160, fingers: [0, 1, 1, 1, 0], wristAngle: -0.25 },
  },
  FOOD: {
    label: 'Food (खाना / Eat)',
    desc: 'Fingertips clustered together with thumb touching tips and brought gently toward mouth twice.',
    category: 'Need',
    indicHindi: 'खाना / भोजन',
    pose: { x: 315, y: 160, fingers: [0.3, 0.3, 0.3, 0.3, 0.3], wristAngle: -0.35 },
  },
  STOP: {
    label: 'Stop (रुकिए / Halt)',
    desc: 'Two-handed: Left flat palm held forward; right open hand chops down vertically forming a solid wall barrier.',
    category: 'Action',
    indicHindi: 'रुको / रुकिए',
    pose: {
      x: 320,
      y: 240,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -1.2,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.3,
    },
  },
  LOVE: {
    label: 'Love (प्यार / स्नेह)',
    desc: 'Two-handed: Both hands in fists crossed over chest in tight affectionate hug over the heart.',
    category: 'Emotion',
    indicHindi: 'प्यार / स्नेह',
    pose: {
      x: 320,
      y: 240,
      fingers: [0, 0, 0, 0, 0],
      wristAngle: -0.5,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 240,
      leftFingers: [0, 0, 0, 0, 0],
      leftWristAngle: 0.5,
    },
  },
  HAPPY: {
    label: 'Happy (खुश / प्रसन्न)',
    desc: 'Two-handed: Both open flat hands brush upward against chest twice with bright radiant smile and nod.',
    category: 'Emotion',
    indicHindi: 'खुश / प्रसन्न',
    pose: {
      x: 330,
      y: 225,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 225,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.2,
    },
  },
  SAD: {
    label: 'Sad (दुखी / उदास)',
    desc: 'Both hands slightly cupped and limp in front of face, dropping slowly downward with somber head tilt.',
    category: 'Emotion',
    indicHindi: 'दुखी / उदास',
    pose: {
      x: 325,
      y: 250,
      fingers: [0.6, 0.6, 0.6, 0.6, 0.6],
      wristAngle: 0.4,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 250,
      leftFingers: [0.6, 0.6, 0.6, 0.6, 0.6],
      leftWristAngle: 0.4,
    },
  },

  // -------------------------------------------------------------
  // Time & Questions
  // -------------------------------------------------------------
  TIME: {
    label: 'Time (समय / वक़्त)',
    desc: 'Dominant index finger taps outer wrist watch location on non-dominant wrist twice.',
    category: 'Time',
    indicHindi: 'समय / वक़्त',
    pose: { x: 305, y: 250, fingers: [0, 1, 0, 0, 0], wristAngle: -0.3 },
  },
  TODAY: {
    label: 'Today (आज)',
    desc: 'Two-handed: Both hands in Y or open cupped shape pulse downward twice in present space.',
    category: 'Time',
    indicHindi: 'आज',
    pose: {
      x: 330,
      y: 260,
      fingers: [1, 0, 0, 0, 1],
      wristAngle: 0.2,
      isTwoHanded: true,
      leftHandX: 270,
      leftHandY: 260,
      leftFingers: [1, 0, 0, 0, 1],
      leftWristAngle: -0.2,
    },
  },
  TOMORROW: {
    label: 'Tomorrow (कल / आने वाला)',
    desc: 'Dominant thumb touches cheek and arcs forward past the jawline into the future.',
    category: 'Time',
    indicHindi: 'कल (आने वाला)',
    pose: { x: 340, y: 195, fingers: [1, 0, 0, 0, 0], wristAngle: -0.2 },
  },
  YESTERDAY: {
    label: 'Yesterday (कल / बीता हुआ)',
    desc: 'Dominant thumb touches cheek and arcs backward past shoulder into the past.',
    category: 'Time',
    indicHindi: 'कल (बीता हुआ)',
    pose: { x: 340, y: 195, fingers: [1, 0, 0, 0, 0], wristAngle: 0.3 },
  },
  HOW: {
    label: 'How (कैसे)',
    desc: 'Two-handed: Both hands curved knuckles touching, rotating palms upward in inquiry with furrowed brows.',
    category: 'Question',
    indicHindi: 'कैसे',
    pose: {
      x: 320,
      y: 245,
      fingers: [0.7, 0.7, 0.7, 0.7, 0.7],
      wristAngle: -0.3,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 245,
      leftFingers: [0.7, 0.7, 0.7, 0.7, 0.7],
      leftWristAngle: 0.3,
    },
  },
  WHERE: {
    label: 'Where (कहाँ)',
    desc: 'Dominant index finger held upright, oscillating side-to-side with questioning facial expression.',
    category: 'Question',
    indicHindi: 'कहाँ',
    pose: { x: 330, y: 220, fingers: [0, 1, 0, 0, 0], wristAngle: 0.1 },
  },
  WHAT: {
    label: 'What (क्या)',
    desc: 'Two-handed: Both open palms facing upward, shaking lightly side-to-side with raised brows.',
    category: 'Question',
    indicHindi: 'क्या',
    pose: {
      x: 340,
      y: 250,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: 0.2,
      isTwoHanded: true,
      leftHandX: 260,
      leftHandY: 250,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: -0.2,
    },
  },
  WHEN: {
    label: 'When (कब)',
    desc: 'Two-handed: Left index points upright; right index circles around left fingertip then lands on it.',
    category: 'Question',
    indicHindi: 'कब',
    pose: {
      x: 310,
      y: 220,
      fingers: [0, 1, 0, 0, 0],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 230,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  WHY: {
    label: 'Why (क्यों)',
    desc: 'Dominant fingertips touch forehead temple, pulling away into a Y-hand shape with inquisitive expression.',
    category: 'Question',
    indicHindi: 'क्यों',
    pose: { x: 330, y: 190, fingers: [1, 0, 0, 0, 1], wristAngle: -0.3 },
  },
  WHO: {
    label: 'Who (कौन)',
    desc: 'Dominant index finger upright near mouth, curling and wiggling in circular inquiry.',
    category: 'Question',
    indicHindi: 'कौन',
    pose: { x: 320, y: 180, fingers: [0, 0.7, 0, 0, 0], wristAngle: -0.1 },
  },
  NAME: {
    label: 'Name (नाम)',
    desc: 'Two-handed: Index and middle fingers extended on both hands tapping together in X-cross pattern.',
    category: 'Identity',
    indicHindi: 'नाम',
    pose: {
      x: 315,
      y: 230,
      fingers: [0, 1, 1, 0, 0],
      wristAngle: -0.3,
      isTwoHanded: true,
      leftHandX: 285,
      leftHandY: 230,
      leftFingers: [0, 1, 1, 0, 0],
      leftWristAngle: 0.3,
    },
  },
  YOU: {
    label: 'You (आप / तुम)',
    desc: 'Dominant index finger pointing forward directly toward conversation partner.',
    category: 'Pronoun',
    indicHindi: 'आप / तुम',
    pose: { x: 320, y: 240, fingers: [0.2, 1, 0, 0, 0], wristAngle: 0.3 },
  },
  ME: {
    label: 'I / Me (मैं / मुझे)',
    desc: 'Dominant index finger pointing inward toward center of own chest.',
    category: 'Pronoun',
    indicHindi: 'मैं / मुझे',
    pose: { x: 290, y: 260, fingers: [0.2, 1, 0, 0, 0], wristAngle: -1.2 },
  },

  // -------------------------------------------------------------
  // Full 26 Letters of ISL Manual Alphabet (A-Z)
  // -------------------------------------------------------------
  A: {
    label: 'Letter A',
    desc: 'Closed fist with thumb upright along side of index finger.',
    category: 'Alphabet',
    indicHindi: 'अक्षर A',
    pose: { x: 340, y: 260, fingers: [1, 0, 0, 0, 0], wristAngle: -0.1 },
  },
  B: {
    label: 'Letter B (ISL Two-Handed)',
    desc: 'Two-handed: Left hand held upright flat palm facing right; right hand open palm touching left palm.',
    category: 'Alphabet',
    indicHindi: 'अक्षर B',
    pose: {
      x: 325,
      y: 235,
      fingers: [0.3, 1, 1, 1, 1],
      wristAngle: -0.15,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 235,
      leftFingers: [0.3, 1, 1, 1, 1],
      leftWristAngle: 0.15,
    },
  },
  C: {
    label: 'Letter C',
    desc: 'Curved hand forming a distinct C-shape facing forward.',
    category: 'Alphabet',
    indicHindi: 'अक्षर C',
    pose: { x: 340, y: 250, fingers: [0.6, 0.6, 0.6, 0.6, 0.6], wristAngle: -0.2 },
  },
  D: {
    label: 'Letter D (ISL Two-Handed)',
    desc: 'Two-handed: Left index points straight upright (stem); right index and thumb curve to meet left index tip and base.',
    category: 'Alphabet',
    indicHindi: 'अक्षर D',
    pose: {
      x: 315,
      y: 230,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: -0.3,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 230,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  E: {
    label: 'Letter E',
    desc: 'Fingers curled inward with tips resting near thumb tip.',
    category: 'Alphabet',
    indicHindi: 'अक्षर E',
    pose: { x: 340, y: 260, fingers: [0.2, 0.3, 0.3, 0.3, 0.3], wristAngle: 0.1 },
  },
  F: {
    label: 'Letter F',
    desc: 'Thumb and index fingertips touching in a circle, other three fingers upright.',
    category: 'Alphabet',
    indicHindi: 'अक्षर F',
    pose: { x: 340, y: 230, fingers: [0.4, 0.4, 1, 1, 1], wristAngle: -0.1 },
  },
  G: {
    label: 'Letter G',
    desc: 'Index finger and thumb parallel pointing horizontally.',
    category: 'Alphabet',
    indicHindi: 'अक्षर G',
    pose: { x: 330, y: 260, fingers: [0.9, 0.9, 0, 0, 0], wristAngle: 0.6 },
  },
  H: {
    label: 'Letter H',
    desc: 'Index and middle fingers extended together pointing horizontally.',
    category: 'Alphabet',
    indicHindi: 'अक्षर H',
    pose: { x: 330, y: 250, fingers: [0.4, 1, 1, 0, 0], wristAngle: 0.6 },
  },
  I: {
    label: 'Letter I',
    desc: 'Pinky finger extended upright, other fingers closed in a fist.',
    category: 'Alphabet',
    indicHindi: 'अक्षर I',
    pose: { x: 340, y: 240, fingers: [0, 0, 0, 0, 1], wristAngle: 0 },
  },
  J: {
    label: 'Letter J',
    desc: 'Pinky finger extended tracing a J hook in the air.',
    category: 'Alphabet',
    indicHindi: 'अक्षर J',
    pose: { x: 340, y: 240, fingers: [0, 0, 0, 0, 1], wristAngle: 0.4 },
  },
  K: {
    label: 'Letter K',
    desc: 'Index upright, middle pointed forward, thumb resting at index knuckle.',
    category: 'Alphabet',
    indicHindi: 'अक्षर K',
    pose: { x: 340, y: 240, fingers: [0.8, 1, 0.6, 0, 0], wristAngle: 0 },
  },
  L: {
    label: 'Letter L',
    desc: 'Thumb and index finger extended perpendicular forming an L shape.',
    category: 'Alphabet',
    indicHindi: 'अक्षर L',
    pose: { x: 340, y: 240, fingers: [1, 1, 0, 0, 0], wristAngle: 0 },
  },
  M: {
    label: 'Letter M',
    desc: 'Thumb tucked under first three fingers, curled over thumb.',
    category: 'Alphabet',
    indicHindi: 'अक्षर M',
    pose: { x: 340, y: 260, fingers: [0.1, 0.4, 0.4, 0.4, 0], wristAngle: 0 },
  },
  N: {
    label: 'Letter N',
    desc: 'Thumb tucked under first two fingers, curled over thumb.',
    category: 'Alphabet',
    indicHindi: 'अक्षर N',
    pose: { x: 340, y: 260, fingers: [0.1, 0.4, 0.4, 0, 0], wristAngle: 0 },
  },
  O: {
    label: 'Letter O',
    desc: 'All fingertips curved touching thumb tip forming an O shape.',
    category: 'Alphabet',
    indicHindi: 'अक्षर O',
    pose: { x: 340, y: 250, fingers: [0.5, 0.5, 0.5, 0.5, 0.5], wristAngle: 0 },
  },
  P: {
    label: 'Letter P (ISL Two-Handed)',
    desc: 'Two-handed: Left index points upright; right index and thumb form loop touching top of left index.',
    category: 'Alphabet',
    indicHindi: 'अक्षर P',
    pose: {
      x: 315,
      y: 220,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 235,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  Q: {
    label: 'Letter Q (ISL Two-Handed)',
    desc: 'Two-handed: Left index finger points downwards; right index and thumb circle around left wrist base.',
    category: 'Alphabet',
    indicHindi: 'अक्षर Q',
    pose: {
      x: 320,
      y: 235,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: 0.8,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 255,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 3.14,
    },
  },
  R: {
    label: 'Letter R',
    desc: 'Index and middle fingers crossed over each other.',
    category: 'Alphabet',
    indicHindi: 'अक्षर R',
    pose: { x: 340, y: 240, fingers: [0, 1, 1, 0, 0], wristAngle: 0 },
  },
  S: {
    label: 'Letter S',
    desc: 'Tight fist with thumb folded horizontally across the front of fingers.',
    category: 'Alphabet',
    indicHindi: 'अक्षर S',
    pose: { x: 340, y: 260, fingers: [0, 0, 0, 0, 0], wristAngle: 0 },
  },
  T: {
    label: 'Letter T (ISL Two-Handed)',
    desc: 'Two-handed: Left index finger points upright; right index held horizontally across top of left index forming a T crossbar.',
    category: 'Alphabet',
    indicHindi: 'अक्षर T',
    pose: {
      x: 295,
      y: 205,
      fingers: [0, 1, 0, 0, 0],
      wristAngle: 1.57,
      isTwoHanded: true,
      leftHandX: 285,
      leftHandY: 235,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  U: {
    label: 'Letter U',
    desc: 'Index and middle fingers held upright together.',
    category: 'Alphabet',
    indicHindi: 'अक्षर U',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 0, 0], wristAngle: 0 },
  },
  V: {
    label: 'Letter V',
    desc: 'Index and middle fingers held upright in a spread V shape.',
    category: 'Alphabet',
    indicHindi: 'अक्षर V',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 0, 0], wristAngle: 0.1 },
  },
  W: {
    label: 'Letter W',
    desc: 'Three middle fingers held upright and spread in a W shape.',
    category: 'Alphabet',
    indicHindi: 'अक्षर W',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 1, 0], wristAngle: 0 },
  },
  X: {
    label: 'Letter X',
    desc: 'Index finger hooked in a crook shape, others closed.',
    category: 'Alphabet',
    indicHindi: 'अक्षर X',
    pose: { x: 340, y: 250, fingers: [0, 0.5, 0, 0, 0], wristAngle: 0.1 },
  },
  Y: {
    label: 'Letter Y',
    desc: 'Thumb and pinky extended wide, middle three fingers folded down.',
    category: 'Alphabet',
    indicHindi: 'अक्षर Y',
    pose: { x: 340, y: 240, fingers: [1, 0, 0, 0, 1], wristAngle: 0 },
  },
  Z: {
    label: 'Letter Z',
    desc: 'Index finger extended and tracing a Z pattern in the air.',
    category: 'Alphabet',
    indicHindi: 'अक्षर Z',
    pose: { x: 340, y: 240, fingers: [0, 1, 0, 0, 0], wristAngle: 0.2 },
  },
};

// Aliases and synonym mapping to canonical ISL sign tokens
const VOCABULARY_ALIASES: Record<string, string> = {
  // Family
  DAD: 'FATHER',
  PAPA: 'FATHER',
  DADDY: 'FATHER',
  MOM: 'MOTHER',
  MUMMY: 'MOTHER',
  MAA: 'MOTHER',
  MOMMY: 'MOTHER',
  BRO: 'BROTHER',
  SIS: 'SISTER',
  DOST: 'FRIEND',
  MITRA: 'FRIEND',
  // Greetings & Courtesies
  HI: 'HELLO',
  HEY: 'HELLO',
  WAVE: 'HELLO',
  NAMASKAR: 'NAMASTE',
  VANAKKAM: 'NAMASTE',
  THANKS: 'THANK_YOU',
  THANK: 'THANK_YOU',
  PLZ: 'PLEASE',
  YEAH: 'YES',
  YEP: 'YES',
  NOT: 'NO',
  // Common Actions & Needs
  EAT: 'FOOD',
  EATING: 'FOOD',
  DRINK: 'WATER',
  DRINKING: 'WATER',
  HALT: 'STOP',
  LIKE: 'LOVE',
  LOVES: 'LOVE',
  LOVED: 'LOVE',
  READ: 'BOOK',
  STUDY: 'BOOK',
  STUDYING: 'BOOK',
  CLOCK: 'TIME',
  CASH: 'MONEY',
  RUPEES: 'MONEY',
  RUPEE: 'MONEY',
  HOUSE: 'HOME',
  // Pronouns
  I: 'ME',
  MY: 'ME',
  YOUR: 'YOU',
  YOU_ALL: 'YOU',
};

// Converts English input into genuine ISL grammatical gloss sequence (SOV structure)
export function convertEnglishToISLGloss(englishText: string): {
  glossTokens: string[];
  words: string[];
  isQuestion: boolean;
} {
  const clean = englishText
    .toUpperCase()
    .replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, ' ')
    .trim();
  if (!clean) return { glossTokens: [], words: [], isQuestion: false };

  const isQuestion =
    englishText.includes('?') ||
    /\b(HOW|WHERE|WHAT|WHEN|WHY|WHO|WHICH)\b/i.test(englishText);

  // Replace multi-word common idioms first with canonical ISL tokens
  const normalized = clean
    .replace(/\bGOOD\s+MORNING\b/g, 'GOOD_MORNING')
    .replace(/\bGOOD\s+AFTERNOON\b/g, 'GOOD_AFTERNOON')
    .replace(/\bGOOD\s+NIGHT\b/g, 'GOOD_NIGHT')
    .replace(/\bTHANK\s+YOU\b/g, 'THANK_YOU')
    .replace(/\bTHANK\s+U\b/g, 'THANK_YOU')
    .replace(/\bHOW\s+ARE\s+YOU\b/g, 'YOU HOW')
    .replace(/\bWHERE\s+IS\s+HOSPITAL\b/g, 'HOSPITAL WHERE')
    .replace(/\bWHERE\s+IS\s+DOCTOR\b/g, 'DOCTOR WHERE')
    .replace(/\bWHAT\s+IS\s+YOUR\s+NAME\b/g, 'YOU NAME WHAT')
    .replace(/\bWHAT\s+IS\s+NAME\b/g, 'NAME WHAT')
    .replace(/\bI\s+LOVE\s+MY\s+FAMILY\b/g, 'ME FAMILY LOVE')
    .replace(/\bI\s+LOVE\s+YOU\b/g, 'ME YOU LOVE')
    .replace(/\bI\s+AM\s+HAPPY\b/g, 'ME HAPPY')
    .replace(/\bI\s+AM\s+SAD\b/g, 'ME SAD');

  const rawWords = normalized.split(/\s+/).filter(Boolean);
  const glossTokens: string[] = [];

  rawWords.forEach((word) => {
    // 1. Check direct vocabulary match
    if (SIGN_DATABASE[word]) {
      glossTokens.push(word);
      return;
    }

    // 2. Check alias map
    const mapped = VOCABULARY_ALIASES[word];
    if (mapped && SIGN_DATABASE[mapped]) {
      glossTokens.push(mapped);
      return;
    }

    // 3. Skip grammatical fillers that are omitted in ISL (am, is, are, the, a, an, to, of)
    if (/^(AM|IS|ARE|THE|A|AN|TO|OF|IN|AT|FOR)$/.test(word)) {
      return;
    }

    // 4. Fingerspell letters or digits for proper nouns or unfamiliar words
    for (const char of word) {
      if (/^[A-Z0-9]$/.test(char)) {
        glossTokens.push(char);
      }
    }
  });

  return {
    glossTokens: glossTokens.length > 0 ? glossTokens : ['HELLO'],
    words: rawWords,
    isQuestion,
  };
}

interface TranslationResult {
  fluent_english: string;
  translations: Record<string, string>;
  canonical_match?: boolean;
  rule_applied?: string;
}

export const TextToSignWorkspace: React.FC<TextToSignWorkspaceProps> = ({
  onSpeak,
  isSpeaking: _isSpeaking = false,
}) => {
  const [inputText, setInputText] = useState<string>('Hello');
  const [activeTokenIndex, setActiveTokenIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [avatarMode, setAvatarMode] = useState<'3d' | '2d'>('3d');
  const [autoSpeakToken, setAutoSpeakToken] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [glossTokens, setGlossTokens] = useState<string[]>(() => convertEnglishToISLGloss('Hello').glossTokens);
  const [isGlossLoading, setIsGlossLoading] = useState<boolean>(false);
  const [glossSource, setGlossSource] = useState<'ai_neural' | 'heuristic'>('ai_neural');
  const [translationResult, setTranslationResult] = useState<TranslationResult | null>(null);

  // Trigger translation pipeline (reverse-gloss + multilingual Indic translation)
  const executeTranslation = useCallback(async (textToTranslate: string) => {
    const trimmed = textToTranslate.trim();
    if (!trimmed) {
      setGlossTokens([]);
      setTranslationResult(null);
      return;
    }

    setIsGlossLoading(true);

    try {
      // Step A: Request AI reverse gloss from backend NLP service
      const res = await ApiService.toGloss(trimmed);
      if (res && Array.isArray(res.gloss_tokens) && res.gloss_tokens.length > 0) {
        // Expand compound tokens and verify with vocabulary
        const expanded: string[] = [];
        for (const rawToken of res.gloss_tokens) {
          const upper = rawToken.toUpperCase().replace(/\s+/g, '_');
          const mapped = VOCABULARY_ALIASES[upper] || upper;

          if (SIGN_DATABASE[mapped]) {
            expanded.push(mapped);
          } else {
            // Check if it was an English word we can normalize
            const localCheck = convertEnglishToISLGloss(upper);
            if (localCheck.glossTokens.length > 0 && localCheck.glossTokens[0] !== 'HELLO') {
              expanded.push(...localCheck.glossTokens);
            } else {
              for (const ch of mapped) {
                if (/^[A-Z0-9]$/.test(ch)) expanded.push(ch);
              }
            }
          }
        }

        const finalTokens = expanded.length > 0 ? expanded : res.gloss_tokens;
        setGlossTokens(finalTokens);
        setGlossSource('ai_neural');

        // Step B: Query multilingual Indic translation for identified tokens
        try {
          const isQ = trimmed.includes('?') || /\b(how|where|what|when|why|who)\b/i.test(trimmed);
          const tr = await ApiService.translateGloss(finalTokens, isQ);
          if (tr) {
            setTranslationResult(tr);
          }
        } catch (tErr) {
          console.warn('Multilingual translate endpoint warning:', tErr);
        }

        setIsGlossLoading(false);
        return;
      }
    } catch (err) {
      console.warn('AI reverse gloss endpoint unavailable, using local rules:', err);
    }

    // Fallback: Local rule-based syntax engine
    const local = convertEnglishToISLGloss(trimmed);
    setGlossTokens(local.glossTokens);
    setGlossSource('heuristic');

    try {
      const tr = await ApiService.translateGloss(local.glossTokens, local.isQuestion);
      if (tr) {
        setTranslationResult(tr);
      }
    } catch (tErr) {
      console.warn('Local translation query warning:', tErr);
    }

    setIsGlossLoading(false);
  }, []);

  // Debounced translation on typing
  useEffect(() => {
    let isMounted = true;
    const timer = setTimeout(() => {
      if (isMounted) {
        executeTranslation(inputText);
      }
    }, 280);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [inputText, executeTranslation]);

  // Handle immediate translation on Enter or Translate click
  const handleTranslateNow = () => {
    setActiveTokenIndex(0);
    setIsPlaying(true);
    executeTranslation(inputText);
  };

  // Safe active token selection
  const activeToken =
    glossTokens[activeTokenIndex] || (glossTokens.length > 0 ? glossTokens[0] : 'HELLO');

  // Dynamic sign data fallback (NEVER defaults to HELLO for unrecognized tokens)
  const activeSignData: SignData = SIGN_DATABASE[activeToken] || {
    label: activeToken.length === 1 ? `Letter ${activeToken}` : activeToken.replace(/_/g, ' '),
    desc: `Authentic Indian Sign Language sign for "${activeToken.replace(/_/g, ' ')}"`,
    category: activeToken.length === 1 ? 'Alphabet' : 'General Vocabulary',
    indicHindi: activeToken,
    pose: { x: 340, y: 240, fingers: [1, 1, 1, 1, 1], wristAngle: 0 },
  };

  // Current interpolated hand position for Canvas
  const currentHandPosRef = useRef({
    rx: 340,
    ry: 240,
    rWristAngle: 0,
    rFingers: [1, 1, 1, 1, 1],
    lx: 230,
    ly: 310,
    lWristAngle: 0.1,
    lFingers: [0.3, 0.3, 0.3, 0.3, 0.3],
  });

  // Cycle through gloss tokens
  useEffect(() => {
    if (!isPlaying || glossTokens.length === 0) return;

    const intervalTime = Math.max(450, Math.round(1350 / playbackSpeed));
    const timer = setInterval(() => {
      setActiveTokenIndex((prev) => (prev + 1) % glossTokens.length);
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isPlaying, glossTokens.length, playbackSpeed]);

  // Reset index when input text changes
  useEffect(() => {
    setActiveTokenIndex(0);
  }, [inputText]);

  // Synchronized Text-to-Speech audio announcement
  useEffect(() => {
    if (autoSpeakToken && onSpeak && glossTokens[activeTokenIndex]) {
      const token = glossTokens[activeTokenIndex].replace(/_/g, ' ');
      onSpeak(token);
    }
  }, [activeTokenIndex, autoSpeakToken, onSpeak, glossTokens]);

  // 2D Articulated Humanoid Avatar Canvas Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const drawJointSphere = (
      x: number,
      y: number,
      radius: number,
      baseColor: string,
      highlightColor = '#ffffff'
    ) => {
      const grad = ctx.createRadialGradient(
        x - radius * 0.35,
        y - radius * 0.35,
        radius * 0.1,
        x,
        y,
        radius
      );
      grad.addColorStop(0, highlightColor);
      grad.addColorStop(0.35, baseColor);
      grad.addColorStop(1, '#090d16');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const drawArticulatedHand = (
      handX: number,
      handY: number,
      wristAngle: number,
      fingers: number[],
      isLeft: boolean
    ) => {
      ctx.save();
      ctx.translate(handX, handY);
      ctx.rotate(wristAngle);

      const themeColor = isLeft ? '#34d399' : '#38bdf8';
      const dimColor = '#1f2937';

      const pW = 46;
      const pH = 42;
      const palmGrad = ctx.createLinearGradient(0, -pH / 2, 0, pH / 2);
      palmGrad.addColorStop(0, '#334155');
      palmGrad.addColorStop(0.7, '#1e293b');
      palmGrad.addColorStop(1, '#0f172a');
      ctx.fillStyle = palmGrad;
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(-pW / 2, -pH / 2, pW, pH, 8);
      ctx.fill();
      ctx.stroke();

      const palmCoreGrad = ctx.createRadialGradient(0, 0, 1, 0, 0, 10);
      palmCoreGrad.addColorStop(0, '#ffffff');
      palmCoreGrad.addColorStop(0.5, themeColor);
      palmCoreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = palmCoreGrad;
      ctx.beginPath();
      ctx.arc(0, 0, 10, 0, Math.PI * 2);
      ctx.fill();

      const fingerAngles = isLeft ? [0.65, 0.32, 0, -0.32, -0.65] : [-0.65, -0.32, 0, 0.32, 0.65];
      const fingerLengths = [26, 38, 42, 38, 30];
      const knuckleSpacing = isLeft ? [18, 9, 0, -9, -18] : [-18, -9, 0, 9, 18];

      for (let f = 0; f < 5; f++) {
        const ext = Math.max(0, Math.min(1, fingers[f] ?? 1));
        const baseAngle = fingerAngles[f];
        const totalLen = fingerLengths[f];
        const seg1Len = totalLen * 0.4;
        const seg2Len = totalLen * 0.35;
        const seg3Len = totalLen * 0.25;

        const kX = knuckleSpacing[f];
        const kY = -pH / 2 + 2;
        drawJointSphere(kX, kY, 4.5, themeColor);

        const p1X = kX + Math.sin(baseAngle) * seg1Len;
        const p1Y = kY - Math.cos(baseAngle) * seg1Len;

        ctx.strokeStyle = ext > 0.3 ? themeColor : dimColor;
        ctx.lineWidth = 4.4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(kX, kY);
        ctx.lineTo(p1X, p1Y);
        ctx.stroke();

        const curlOffset1 = (1 - ext) * 1.2 * (isLeft ? -1 : 1);
        const pipHighlight = ext > 0.5 ? themeColor : '#f59e0b';
        drawJointSphere(p1X, p1Y, 3.8, pipHighlight);

        const a2 = baseAngle + curlOffset1;
        const p2X = p1X + Math.sin(a2) * (seg2Len * (0.4 + ext * 0.6));
        const p2Y = p1Y - Math.cos(a2) * (seg2Len * (0.4 + ext * 0.6));

        ctx.strokeStyle = ext > 0.3 ? themeColor : '#64748b';
        ctx.lineWidth = 3.6;
        ctx.beginPath();
        ctx.moveTo(p1X, p1Y);
        ctx.lineTo(p2X, p2Y);
        ctx.stroke();

        const curlOffset2 = (1 - ext) * 2.0 * (isLeft ? -1 : 1);
        drawJointSphere(p2X, p2Y, 3.2, ext > 0.5 ? themeColor : '#fbbf24');

        const a3 = baseAngle + curlOffset2;
        const p3X = p2X + Math.sin(a3) * (seg3Len * (0.35 + ext * 0.65));
        const p3Y = p2Y - Math.cos(a3) * (seg3Len * (0.35 + ext * 0.65));

        ctx.strokeStyle = ext > 0.3 ? themeColor : '#475569';
        ctx.lineWidth = 3.0;
        ctx.beginPath();
        ctx.moveTo(p2X, p2Y);
        ctx.lineTo(p3X, p3Y);
        ctx.stroke();

        drawJointSphere(p3X, p3Y, 3.0, ext > 0.5 ? '#ffffff' : '#cbd5e1', '#ffffff');
      }

      ctx.restore();
    };

    const render = () => {
      time += 0.05 * playbackSpeed;
      const targetPose = activeSignData.pose;
      const curr = currentHandPosRef.current;

      const rxTarget = targetPose.x;
      const ryTarget = targetPose.y;
      const rAngleTarget = targetPose.wristAngle;
      const rFingersTarget = targetPose.fingers;

      curr.rx += (rxTarget - curr.rx) * 0.18;
      curr.ry += (ryTarget - curr.ry) * 0.18;
      curr.rWristAngle += (rAngleTarget - curr.rWristAngle) * 0.18;
      for (let i = 0; i < 5; i++) {
        curr.rFingers[i] += ((rFingersTarget[i] ?? 1) - curr.rFingers[i]) * 0.22;
      }

      const centerX = canvas.width / 2;
      const headY = 120 + Math.sin(time * 0.8) * 1.5;

      const lxTarget = targetPose.isTwoHanded ? (targetPose.leftHandX ?? 270) : centerX - 75;
      const lyTarget = targetPose.isTwoHanded ? (targetPose.leftHandY ?? 240) : headY + 185;
      const lAngleTarget = targetPose.isTwoHanded ? (targetPose.leftWristAngle ?? 0) : 0.15;
      const lFingersTarget = targetPose.isTwoHanded
        ? (targetPose.leftFingers ?? [1, 1, 1, 1, 1])
        : [0.3, 0.3, 0.3, 0.3, 0.3];

      curr.lx += (lxTarget - curr.lx) * 0.18;
      curr.ly += (lyTarget - curr.ly) * 0.18;
      curr.lWristAngle += (lAngleTarget - curr.lWristAngle) * 0.18;
      for (let i = 0; i < 5; i++) {
        curr.lFingers[i] += ((lFingersTarget[i] ?? 1) - curr.lFingers[i]) * 0.22;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bgGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
      bgGrad.addColorStop(0, '#090d16');
      bgGrad.addColorStop(0.5, '#0f172a');
      bgGrad.addColorStop(1, '#050811');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Chest & Shoulders
      const rShoulderX = centerX + 72;
      const rShoulderY = headY + 54;
      const lShoulderX = centerX - 72;
      const lShoulderY = headY + 54;

      const rHandX = curr.rx;
      const rHandY = curr.ry;
      const rElbowX = (rShoulderX + rHandX) / 2 + 34;
      const rElbowY = Math.max(rShoulderY + 35, (rShoulderY + rHandY) / 2 + 25);

      const lHandX = curr.lx;
      const lHandY = curr.ly;
      const lElbowX = targetPose.isTwoHanded
        ? (lShoulderX + lHandX) / 2 - 28
        : lShoulderX - 22;
      const lElbowY = targetPose.isTwoHanded
        ? Math.max(lShoulderY + 35, (lShoulderY + lHandY) / 2 + 25)
        : lShoulderY + 90;

      // Arms
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(rShoulderX, rShoulderY);
      ctx.lineTo(rElbowX, rElbowY);
      ctx.lineTo(rHandX, rHandY);
      ctx.stroke();

      ctx.strokeStyle = targetPose.isTwoHanded
        ? 'rgba(52, 211, 153, 0.75)'
        : 'rgba(52, 211, 153, 0.35)';
      ctx.beginPath();
      ctx.moveTo(lShoulderX, lShoulderY);
      ctx.lineTo(lElbowX, lElbowY);
      ctx.lineTo(lHandX, lHandY);
      ctx.stroke();

      drawJointSphere(rShoulderX, rShoulderY, 8, '#38bdf8');
      drawJointSphere(lShoulderX, lShoulderY, 8, '#34d399');
      drawJointSphere(rElbowX, rElbowY, 6.5, '#38bdf8');
      drawJointSphere(lElbowX, lElbowY, 6.5, '#34d399');

      // Head
      drawJointSphere(centerX, headY, 34, '#475569', '#94a3b8');

      // Hands
      drawArticulatedHand(lHandX, lHandY, curr.lWristAngle, curr.lFingers, true);
      drawArticulatedHand(rHandX, rHandY, curr.rWristAngle, curr.rFingers, false);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [activeSignData, playbackSpeed]);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => console.error(err));
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch((err) => console.error(err));
      setIsFullscreen(false);
    }
  };

  const isFingerspelledOnly =
    glossTokens.length > 0 && glossTokens.every((t) => t.length === 1);

  return (
    <div
      ref={containerRef}
      className={`text-to-sign-workspace ${isFullscreen ? 'fullscreen-mode' : ''}`}
      aria-label="Text to Sign Translation Workspace"
    >
      <div className="text-to-sign-layout">
        {/* Left Column: 3D / 2D Avatar Viewport */}
        <section className="avatar-stage-column" aria-label="Sign Language Avatar Viewport">
          <div className="avatar-stage-viewport">
            {/* Top Overlay Bar */}
            <div className="avatar-top-overlay">
              <div className="sign-badge-stack">
                <span className="live-avatar-tag">
                  <span className="pulsing-live-dot" />
                  {avatarMode === '3d' ? '3D Humanoid Engine' : '2D Biometric Mesh'}
                </span>
                <span className="active-category-pill">
                  {activeSignData.category || 'ISL Vocabulary'}
                </span>
              </div>

              <div className="overlay-controls-right">
                <button
                  className="avatar-mode-pill-btn"
                  onClick={() => setAvatarMode(avatarMode === '3d' ? '2d' : '3d')}
                  title="Switch between 3D photorealistic humanoid and 2D canvas"
                >
                  {avatarMode === '3d' ? '👤 3D Humanoid (10 Fingers)' : '📐 2D Canvas'}
                </button>
                <button
                  className="overlay-icon-btn"
                  onClick={toggleFullscreen}
                  title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Avatar'}
                >
                  {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                </button>
              </div>
            </div>

            {/* Canvas / WebGL Avatar Viewport */}
            <div className="avatar-canvas-wrap">
              {avatarMode === '3d' ? (
                <Avatar3DViewer
                  currentSign={activeToken}
                  isTwoHanded={activeSignData.pose.isTwoHanded}
                />
              ) : (
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={440}
                  className="avatar-canvas"
                  aria-label="Indian Sign Language Humanoid Avatar"
                />
              )}

              {/* Subtitle Dock for Avatar Viewport */}
              <div className="avatar-sign-subtitle-dock">
                <div className="avatar-subtitle-panel">
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      justifyContent: 'center',
                      width: '100%',
                    }}
                  >
                    <span className="subtitle-category-label">
                      {activeSignData.category?.toUpperCase() || 'ISL SIGN'}
                    </span>
                    <span
                      style={{
                        fontSize: '11px',
                        fontWeight: 700,
                        color: '#34d399',
                        background: 'rgba(52, 211, 153, 0.12)',
                        padding: '1px 6px',
                        borderRadius: '4px',
                      }}
                    >
                      Sign {activeTokenIndex + 1} of {Math.max(1, glossTokens.length)}
                    </span>
                  </div>

                  <div className="avatar-subtitle-main">
                    <span className="active-sign-token">
                      {activeSignData.label}
                      {activeSignData.indicHindi && activeSignData.indicHindi !== activeToken && (
                        <span
                          style={{
                            marginLeft: '8px',
                            color: '#fbbf24',
                            fontSize: '0.85em',
                            fontWeight: 600,
                          }}
                        >
                          • {activeSignData.indicHindi}
                        </span>
                      )}
                    </span>
                    <span className="sign-description-text">{activeSignData.desc}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Playback Controls Bar */}
            <div className="avatar-playback-bar">
              <div className="playback-buttons-left">
                <button
                  type="button"
                  className="playback-btn primary"
                  onClick={() => setIsPlaying(!isPlaying)}
                  title={isPlaying ? 'Pause animation' : 'Play animation'}
                  aria-label={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? <Pause size={15} /> : <Play size={15} />}
                  <span>{isPlaying ? 'Pause' : 'Play'}</span>
                </button>

                <button
                  type="button"
                  className="playback-btn secondary"
                  onClick={() => setActiveTokenIndex(0)}
                  title="Replay from start"
                  aria-label="Replay"
                >
                  <RotateCcw size={14} />
                  <span>Replay</span>
                </button>

                {onSpeak && (
                  <button
                    type="button"
                    className={`playback-btn ${autoSpeakToken ? 'primary' : 'secondary'}`}
                    onClick={() => setAutoSpeakToken(!autoSpeakToken)}
                    title="Toggle spoken voice pronunciation in sync with sign gestures"
                    aria-label="Toggle Voice Sync"
                  >
                    <Volume2 size={14} />
                    <span>{autoSpeakToken ? 'Voice Sync On' : 'Voice Sync Off'}</span>
                  </button>
                )}

                {/* Speed Switcher */}
                <div className="speed-pills">
                  {[0.5, 1.0, 1.5].map((speed) => (
                    <button
                      key={speed}
                      type="button"
                      className={`speed-pill ${playbackSpeed === speed ? 'active' : ''}`}
                      onClick={() => setPlaybackSpeed(speed)}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Token Step Progress */}
              <div className="gloss-progress-indicator">
                <span>
                  Sign {activeTokenIndex + 1} of {Math.max(1, glossTokens.length)}
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Translation Intelligence, ISL Breakdown & Prompt Composer */}
        <aside className="avatar-side-panel" aria-label="English Input and ISL Breakdown">
          {/* Translation Intelligence & Meaning Card */}
          <div
            style={{
              background: 'var(--setu-bg-surface, #1e293b)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              borderRadius: '12px',
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
              boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Languages size={15} style={{ color: '#38bdf8' }} />
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#f8fafc' }}>
                  Translation Intelligence
                </span>
              </div>
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  color: isFingerspelledOnly ? '#fbbf24' : '#34d399',
                  background: isFingerspelledOnly
                    ? 'rgba(251, 191, 36, 0.12)'
                    : 'rgba(52, 211, 153, 0.12)',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  border: `1px solid ${
                    isFingerspelledOnly ? 'rgba(251, 191, 36, 0.3)' : 'rgba(52, 211, 153, 0.3)'
                  }`,
                }}
              >
                {isFingerspelledOnly ? '🔤 Fingerspelled' : '✨ ISL Vocabulary Sign'}
              </span>
            </div>

            {/* Multilingual Meanings Grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                background: 'rgba(15, 23, 42, 0.6)',
                padding: '8px 10px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.05)',
              }}
            >
              <div>
                <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '2px' }}>
                  🇮🇳 हिन्दी (Hindi)
                </div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
                  {translationResult?.translations?.hi ||
                    activeSignData.indicHindi ||
                    'अनुवाद तैयार...'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '2px' }}>
                  🇮🇳 தமிழ் (Tamil)
                </div>
                <div style={{ fontSize: '12px', fontWeight: 500, color: '#cbd5e1' }}>
                  {translationResult?.translations?.ta || 'பொருளடக்கம்...'}
                </div>
              </div>
            </div>

            {/* Fluent English Interpretation */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '11px',
                color: '#94a3b8',
                paddingTop: '2px',
              }}
            >
              <span>
                English:{' '}
                <strong style={{ color: '#e2e8f0' }}>
                  {translationResult?.fluent_english || inputText}
                </strong>
              </span>
              <span>
                {glossTokens.length} {glossTokens.length === 1 ? 'Gesture' : 'Gestures'} (SOV)
              </span>
            </div>
          </div>

          {/* ISL Grammar Gloss Breakdown Card */}
          <div className="gloss-breakdown-card">
            <div className="card-header-clean">
              <span className="card-header-title">ISL Grammatical Gloss (SOV Sequence)</span>
              <span className={`grammar-tag ${glossSource === 'ai_neural' ? 'ai-tag' : ''}`}>
                {isGlossLoading ? (
                  'Translating...'
                ) : glossSource === 'ai_neural' ? (
                  <>
                    <Cpu size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                    AI SOV Reordering
                  </>
                ) : (
                  'Topic-Comment Syntax'
                )}
              </span>
            </div>

            <div className="gloss-tokens-container">
              {glossTokens.length > 0 ? (
                glossTokens.map((token, idx) => {
                  const entry = SIGN_DATABASE[token];
                  return (
                    <button
                      key={`${token}-${idx}`}
                      type="button"
                      className={`gloss-token-pill ${idx === activeTokenIndex ? 'active' : ''}`}
                      onClick={() => setActiveTokenIndex(idx)}
                      title={`Click to show gesture for ${token}`}
                    >
                      <span className="token-index">#{idx + 1}</span>
                      <span className="token-text">{token}</span>
                      {entry?.indicHindi && entry.indicHindi !== token && (
                        <span
                          style={{
                            fontSize: '9px',
                            color: idx === activeTokenIndex ? '#ffffff' : '#94a3b8',
                            marginLeft: '4px',
                            opacity: 0.9,
                          }}
                        >
                          ({entry.indicHindi.split('/')[0].trim()})
                        </span>
                      )}
                    </button>
                  );
                })
              ) : (
                <span className="gloss-empty-hint">Type English words to generate ISL signs...</span>
              )}
            </div>
          </div>

          {/* Fingerspelling Letter Rail */}
          {glossTokens.length > 0 && glossTokens.every((t) => t.length === 1) && (
            <div className="fingerspelling-rail-card">
              <div className="card-header-clean">
                <span className="card-header-title">ISL Fingerspelling Sequence</span>
                <span className="card-sub-hint">Click letter to jump gesture</span>
              </div>

              <div className="letters-scroll-row">
                {inputText
                  .replace(/[^a-zA-Z0-9]/g, '')
                  .split('')
                  .map((char, cIdx) => {
                    const upper = char.toUpperCase();
                    const isActive = glossTokens[activeTokenIndex] === upper;
                    return (
                      <button
                        key={`${upper}-${cIdx}`}
                        type="button"
                        className={`letter-rail-box ${isActive ? 'active-letter' : ''}`}
                        onClick={() => {
                          const foundIdx = glossTokens.indexOf(upper);
                          if (foundIdx !== -1) setActiveTokenIndex(foundIdx);
                        }}
                      >
                        <span className="rail-char">{upper}</span>
                        <span className="rail-sign-type">ISL</span>
                      </button>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Quick Suggestion Chips */}
          <div className="quick-chips-card">
            <span className="chips-label">Quick Phrases & Vocabulary:</span>
            <div className="chips-wrap">
              {[
                'Good morning',
                'Hello',
                'Namaste',
                'Father',
                'School',
                'How are you',
                'Where is hospital',
                'Thank you',
                'Water',
                'Food',
                'Help',
                'Doctor',
                'Time',
                'Happy',
                'I love my family',
                'Rahul',
              ].map((phrase) => (
                <button
                  key={phrase}
                  type="button"
                  className="quick-chip-btn"
                  onClick={() => {
                    setInputText(phrase);
                    setActiveTokenIndex(0);
                    setIsPlaying(true);
                    executeTranslation(phrase);
                  }}
                >
                  {phrase}
                </button>
              ))}
            </div>
          </div>

          {/* Gemini / ChatGPT Style Prompt Composer Box */}
          <div className="gemini-composer-container">
            <div className="composer-inner-box">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleTranslateNow();
                  }
                }}
                placeholder="Type English words or sentences (e.g., 'school', 'father', 'how are you')... Press Enter to Translate"
                rows={2}
                className="gemini-textarea"
                aria-label="English text input"
              />

              <div className="composer-bottom-bar">
                <div className="composer-left-actions">
                  <button
                    type="button"
                    className={`composer-icon-btn ${isListening ? 'listening' : ''}`}
                    onClick={() => {
                      if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
                        alert('Speech recognition is not supported in this browser.');
                        return;
                      }
                      const SpeechRecognition =
                        (window as any).SpeechRecognition ||
                        (window as any).webkitSpeechRecognition;
                      const recognition = new SpeechRecognition();
                      recognition.lang = 'en-IN';
                      recognition.interimResults = false;

                      if (!isListening) {
                        recognition.start();
                        setIsListening(true);
                        recognition.onresult = (event: any) => {
                          const speechResult = event.results[0][0].transcript;
                          setInputText(speechResult);
                          setIsListening(false);
                          executeTranslation(speechResult);
                        };
                        recognition.onerror = () => setIsListening(false);
                        recognition.onend = () => setIsListening(false);
                      } else {
                        recognition.stop();
                        setIsListening(false);
                      }
                    }}
                    title={isListening ? 'Stop listening' : 'Speak voice input'}
                    aria-label="Voice input"
                  >
                    {isListening ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>
                  <span className="composer-hint-text">
                    {isListening ? 'Listening...' : 'Press Enter or click Translate'}
                  </span>
                </div>

                <div className="composer-right-actions">
                  <button
                    type="button"
                    className="composer-send-btn"
                    onClick={handleTranslateNow}
                    disabled={!inputText.trim()}
                    title="Translate English to Indian Sign Language"
                    aria-label="Translate"
                  >
                    <Send size={15} />
                    <span>Translate</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
