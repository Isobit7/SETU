# 05. Top 15 Viva & Examination Questions (With Model Answers)

*Review these 15 questions before your presentation or external viva examination. These answers are tailored to highlight the engineering rigor of SETU.*

---

### Q1: "Why did you use MediaPipe instead of training a Convolutional Neural Network (CNN) directly on RGB video frames?"
**Model Answer:**
> *"Sir, training a 3D-CNN directly on raw video frames makes the model learn the background, room lighting, skin tone, and clothing rather than sign geometry. It also requires heavy GPUs and tens of gigabytes of storage. By using Google MediaPipe Holistic as our vision backbone, we decouple feature extraction from classification. MediaPipe gives us 543 spatial coordinates in 12ms. This guarantees 100% user privacy (no images are saved), makes the model immune to lighting or skin tone variations, and reduces inference time from 200ms down to under 2ms, allowing it to run smoothly on standard laptops without expensive hardware."*

---

### Q2: "What is the difference between ASL and ISL, and why can't we just use existing ASL models?"
**Model Answer:**
> *"Sir, American Sign Language (ASL) is almost entirely single-handed for the alphabet, whereas Indian Sign Language (ISL) is predominantly dual-handed for letters like B, D, P, T, and Q, as well as common everyday phrases. Furthermore, ISL follows Subject-Object-Verb (SOV) grammatical syntax (similar to Hindi and Tamil), whereas ASL and English follow Subject-Verb-Object (SVO). If you run an ASL model on an Indian signer, it completely fails to recognize two-handed contact geometry."*

---

### Q3: "How do you handle the problem of users sitting at different distances from the webcam?"
**Model Answer:**
> *"Sir, we implemented Canonical Coordinate Normalization. For every hand detected, we shift Landmark 0 (the wrist) to the origin (0, 0, 0). Then, we calculate the Euclidean distance between the wrist and the middle finger knuckle (Landmark 9) and divide every coordinate by this bone length. This makes the feature vector scale-invariant and position-invariant: the model sees the exact same normalized geometry whether the signer is sitting 0.5 meters or 2 meters from the camera."*

---

### Q4: "Why did you choose WebSockets instead of a traditional REST API for sending video frames?"
**Model Answer:**
> *"Sir, HTTP REST requires opening a new TCP handshake and sending large HTTP headers with every single frame. At 30 frames per second, this introduces 150 to 250 milliseconds of latency per request, causing unacceptable video stutter. WebSockets provide a persistent, full-duplex TCP connection with only 2 bytes of frame overhead. This reduces our round-trip transmission latency to under 15 milliseconds, achieving a smooth 30 FPS real-time translation experience."*

---

### Q5: "What is your BiLSTM architecture and why is it needed over a simple MLP?"
**Model Answer:**
> *"Sir, an MLP is ideal for static alphabet signs where a single snapshot is sufficient. But continuous sign language contains motion over time (for example, the signs for 'Hello' or 'Thank you' have movement trajectories). Our BiLSTM uses a temporal sliding window of 30 frames (1 second). Because it is bidirectional, it evaluates both the forward movement and the backward contextual transition of the hands, allowing it to distinguish signs that share similar ending postures but have different starting gestures with 94.2% test accuracy."*

---

### Q6: "How did you solve the problem of camera jitter or shaking hands?"
**Model Answer:**
> *"Sir, we implemented a 1-Euro adaptive low-pass filter on all 42 hand landmarks. Traditional filters either cause high lag during fast movements or fail to eliminate jitter at rest. The 1-Euro filter dynamically adjusts its cutoff frequency based on landmark velocity: during rapid hand movements, it lets raw motion pass through with zero lag, but when the hand rests or holds a sign, it aggressively dampens jitter."*

---

### Q7: "What happens when the user moves their hands from one sign to another? Doesn't the model falsely detect intermediate letters?"
**Model Answer:**
> *"Sir, that is known as transitional movement artifact (PROB-09). We solved it with two techniques: First, Kinetic Velocity Gating, which detects when the wrist's speed exceeds a threshold and pauses classification during fast sweeps. Second, Hysteresis Debouncing, which requires the neural network to output the exact same sign with over 50% confidence for at least 3 consecutive frames before committing it to the text buffer."*

---

### Q8: "How does the NLP module translate ISL Gloss into fluent English and regional languages?"
**Model Answer:**
> *"Sir, ISL signers communicate in 'gloss' (e.g., [STORE] [I] [GO] [WILL]). In English, this must be re-ordered to 'I will go to the store'. Our NLP pipeline uses a Syntax Transformer that performs rule-based and neural dependency reordering. It also includes an Indic Language Transformer that maps the tokens into Hindi, Tamil, Telugu, and Kannada in native SOV grammar."*

---

### Q9: "What are the latest weights in the project and what are their dates?"
**Model Answer:**
> *"Sir, our production weights were trained on November 6, 2026. The static 26-class alphabet model (`realsign_26class_mlp.pth`) achieves 96.8% test accuracy, the Amrita 100-phrase sequence model (`amrita_isl_phrase_bilstm.pth`) achieves 94.2% accuracy, and the ONNX edge models for Hindi vowels and Numerals achieve 98.1% and 97.4% accuracy, respectively."*

---

### Q10: "Can this system work if there is no internet connection?"
**Model Answer:**
> *"Yes, Sir! We exported our PyTorch neural networks into ONNX format and embedded them inside the browser's public assets (`frontend/public/models/`). When offline, the frontend uses ONNX Runtime Web via WebAssembly, allowing complete real-time alphabet recognition directly inside the client's browser with zero server connection."*

---

### Q11: "How do you distinguish between two-handed signs like 'B' and single-handed signs like 'D'?"
**Model Answer:**
> *"Sir, in PROB-02 we discovered that processing hands independently erased inter-hand distance. We fixed this by augmenting our feature vector with 21 Euclidean distance scalars connecting corresponding joints between both hands ($||L_{left, i} - L_{right, i}||_2$) and the relative 3D vector between both wrists. This makes two-handed contact geometry an explicit feature."*

---

### Q12: "How does your 3D Hand Visualizer in Practice Studio work?"
**Model Answer:**
> *"Sir, in `Practice3DHandViewer.tsx`, we defined true 3D spatial joint coordinates for anatomical hand postures. We project these coordinates onto a 2D canvas using 3D perspective projection matrices. Users can click and drag to orbit around the hand in 3D (yaw and pitch rotation) and select camera presets such as Side Profile or Signer POV to inspect finger curvature."*

---

### Q13: "What is the role of Non-Manual Markers (facial expressions) in SETU?"
**Model Answer:**
> *"Sir, in sign language, facial expressions change the grammatical meaning of a sentence. For instance, raising eyebrows transforms a statement into a question ('Are you coming?'), while a head nod confirms affirmation. We track 468 facial mesh landmarks from MediaPipe Holistic to detect head nod velocity and eyebrow displacement in real time."*

---

### Q14: "What dataset did you train on?"
**Model Answer:**
> *"Sir, we utilized the RealSign ISL dataset consisting of 7,545 multi-signer alphabet images, the Amrita Continuous SLR video dataset with 978 phrase sequences, as well as dedicated Hindi Vowel and Numeral datasets. We also injected 4,000 synthetic 'no_sign' resting vectors to eliminate false positives."*

---

### Q15: "What are your future enhancements for this project?"
**Model Answer:**
> *"Sir, our future roadmap includes: 1) Expanding domain-specific vocabularies for legal and emergency services, 2) Supporting localized Indian sign dialects (such as Southern vs Northern ISL variations), and 3) Integrating generative 3D full-body avatars that translate spoken doctor instructions back into sign language for deaf patients."*
