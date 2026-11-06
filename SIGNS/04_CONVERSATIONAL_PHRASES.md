# SETU — Dynamic Conversational Phrases & Words Catalog

Unlike static fingerspelling, conversational words and phrases in Indian Sign Language (ISL) involve **spatio-temporal motion trajectories**, multi-joint body anchoring, and non-manual facial markers.

SETU hosts two production models for dynamic phrases:
1. **`amrita_isl_phrase_bilstm`:** 49 authentic continuous phrase categories from 978 native ISL video sequences (**78.57% Top-1, 90.82% Top-3**).
2. **`isl_sequence_lstm`:** 7 core high-frequency conversational phrases with 30-frame temporal attention (**94.20% Accuracy**).

---

## 🗣️ The 49 Authentic Conversational Phrases (Amrita SLR Dataset)

The 49 phrases are organized by practical everyday communicative domain:

### 1. Greetings & Social Courtesies
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `hello` | **HELLO** | Open right palm waving outward at temple level. | `"Hello!"` / `"Greetings"` |
| `good_morning` | **GOOD MORNING** | Right hand touches chin ("good"), then both hands rise outward like the sun ("morning"). | `"Good morning."` |
| `thank_you` | **THANK YOU** | Flat right hand touches chin/lips, then extends forward towards the viewer. | `"Thank you very much."` |
| `welcome` | **WELCOME** | Both open hands held at chest level curving inward toward body. | `"You are welcome."` |
| `sorry` | **SORRY** | Right fist makes small circular rubbing motions over center of chest. | `"I am sorry."` |
| `you_are_perfect` | **YOU ARE PERFECT** | Right index points forward ("you"), then "OK" circle sign held decisively with nod. | `"You are doing great / perfect!"` |

### 2. Emergency, Safety & Civic Services
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `help` | **HELP** | Flat left palm acts as platform; closed right fist with thumb up rests on it and moves upward together. | `"Please help me."` |
| `hospital` | **HOSPITAL** | Right index finger traces a cross (`+`) on upper left shoulder/bicep. | `"Hospital"` |
| `doctor` | **DOCTOR** | Right index and middle fingers tap inside left wrist (taking pulse). | `"Doctor"` |
| `police` | **POLICE** | Two fingers tap right shoulder (indicating shoulder badge). | `"Police"` |
| `ambulance` | **AMBULANCE** | Right hand mimics flashing emergency beacon light spinning above head. | `"Emergency ambulance needed."` |
| `power` | **POWER / ELECTRICITY** | Both index fingers touch tips together like sparking wires. | `"Power / Electricity"` |
| `stop` | **STOP** | Flat vertical right hand chops down firmly onto open horizontal left palm. | `"Please stop."` |

### 3. Family & Relationships
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `mother` | **MOTHER** | Right thumb touches side of chin or nose ring location. | `"Mother"` |
| `father` | **FATHER** | Right index finger touches right mustache area or forehead. | `"Father"` |
| `brother` | **BROTHER** | Sign for "boy" followed by index fingers hooked side-by-side. | `"Brother"` |
| `friend` | **FRIEND** | Both index fingers hook together like linked chain links. | `"Friend"` |
| `boy` | **BOY** | Right hand mimics pulling the brim of a cap at the forehead. | `"Boy"` |
| `girl` | **GIRL** | Right index finger traces line along right jawline/cheek. | `"Girl"` |
| `baby` | **BABY** | Both arms crossed in front of chest rocking back and forth. | `"Baby"` |

### 4. Basic Daily Needs & Actions
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `eat` | **EAT** | Clustered fingertips of right hand move toward mouth repeatedly. | `"I want food / eat."` |
| `drinking` | **DRINKING / WATER** | Right hand in "C" cup or "W" shape tips back near mouth. | `"I would like water to drink."` |
| `sleep` | **SLEEP** | Both palms pressed together resting beside tilted right cheek. | `"I am sleepy / sleep."` |
| `sit` | **SIT** | Two fingers of right hand drape over two fingers of left hand like legs on a bench. | `"Please sit down."` |
| `stand` | **STAND** | Right index and middle "legs" stand upright on flat left palm. | `"Please stand up."` |
| `work` | **WORK** | Right wrist taps down onto left wrist repeatedly with closed fists. | `"Work"` |
| `study` | **STUDY** | Open right palm sweeps back and forth over flat open left palm (reading). | `"Study"` |
| `student` | **STUDENT** | Sign for "study" followed by both hands moving down torso ("person"). | `"Student"` |
| `teacher` | **TEACHER** | Flattened hands at temples moving forward ("knowledge") + person suffix. | `"Teacher"` |
| `write` | **WRITE** | Right thumb and index mimic holding a pen writing across left palm. | `"Write"` |

### 5. Time & Calendar Indicators
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `today` | **TODAY** | Both "Y" hands move downward together in front of body. | `"Today"` |
| `tomorrow` | **TOMORROW** | Right thumb moves forward from right cheek in an arc. | `"Tomorrow"` |
| `yesterday` | **YESTERDAY** | Right thumb moves backward from cheek toward ear. | `"Yesterday"` |
| `afternoon` | **AFTERNOON** | Flat right forearm rests on left hand; right arm at $45^\circ$ angle (sun setting). | `"In the afternoon"` |
| `evening` | **EVENING** | Right hand bends down over left wrist (sun sinking). | `"In the evening"` |
| `night` | **NIGHT** | Both curved hands cross in front of chest (darkness falling). | `"Tonight / Night"` |

### 6. Descriptors & Common Items
| Phrase Key | Display Title | Gestural Trajectory & Body Anchor | English Translation Output |
|---|---|---|---|
| `good` | **GOOD** | Flat right hand moves forward from lips with gentle nod. | `"Good"` |
| `bad` | **BAD** | Flat hand placed at lips then turned downward with head shake. | `"Bad"` |
| `angry` | **ANGRY** | Clawed right hand pulls upward in front of face with furrowed brows. | `"Angry"` |
| `name` | **NAME** | "H" fingers of both hands tap each other crosswise at chest. | `"Name"` |
| `teeth` | **TEETH** | Index finger points to smile/teeth. | `"Teeth"` |
| `nose` | **NOSE** | Index finger touches tip of nose. | `"Nose"` |
| `hand` | **HAND** | Right hand pats the back of left hand. | `"Hand"` |
| `ring` | **RING** | Right index slips an imaginary ring onto left ring finger. | `"Ring"` |
| `umbrella` | **UMBRELLA** | Both fists stacked, right hand lifts upward (opening umbrella). | `"Umbrella"` |
| `monsoon` | **MONSOON / RAIN** | Both clawed hands flutter fingers downward like falling raindrops. | `"Monsoon / Rainy season"` |
| `cat` | **CAT** | Both hands mimic pulling whiskers outward from cheeks. | `"Cat"` |
| `dog` | **DOG** | Right hand snaps fingers and pats right thigh. | `"Dog"` |
| `bird` | **BIRD** | Thumb and index at mouth opening and closing like a bird's beak. | `"Bird"` |
| `cow` | **COW** | "Y" hands at temples pointing outward like cow horns. | `"Cow"` |
| `fish` | **FISH** | Flat right hand weaves forward like a fish swimming. | `"Fish"` |
| `black` | **BLACK** | Right index finger rubs right eyebrow or black hair. | `"Black"` |
| `grey` | **GREY** | Both open hands with fingers sliding between each other (mixing colors). | `"Grey"` |

---

## ⏱️ Dynamic Gesture Segmentation: Onset & Latching

To prevent background noise or casual movements from triggering false phrases:
1. **Kinetic Energy Threshold ($E_k > 0.04$):** Gesture onset is detected only when cumulative hand landmark velocity exceeds the baseline threshold.
2. **Phase Progression:** Preparation $\to$ Stroke $\to$ Settlement.
3. **Streak Debouncing:** The model requires a **4-consecutive-frame prediction streak** before latching the word into the SentenceAccumulator.

