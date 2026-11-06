# SETU — AI4Bharat INCLUDE Large-Vocabulary Signs (263 Classes)

SETU integrates the official **AI4Bharat INCLUDE** benchmark model (`models/pretrained/include_no_cnn_lstm.pth`), which spans **263 distinct signs across 15 semantic domains** recorded by native Deaf signers across multiple national training centers in India.

---

## 📂 Semantic Domains & Vocabulary Breakdown

### 1. People, Family & Professions (34 Signs)
`actor`, `adult`, `artist`, `author`, `baby`, `blind`, `boy`, `brother`, `child`, `cousin`, `daughter`, `deaf`, `doctor`, `enemy`, `family`, `father`, `female`, `friend`, `girl`, `grandfather`, `grandmother`, `king`, `lawyer`, `male`, `man`, `mother`, `neighbor`, `nurse`, `police`, `president`, `priest`, `sister`, `son`, `teacher`

### 2. Emergency, Health & Anatomy (28 Signs)
`accident`, `alive`, `attack`, `bathroom`, `bleed`, `blind`, `blood`, `body`, `cough`, `cure`, `danger`, `deaf`, `dead`, `disease`, `doctor`, `emergency`, `fever`, `head`, `heart`, `hospital`, `hurt`, `medicine`, `nurse`, `pain`, `patient`, `poison`, `sick`, `vomit`

### 3. Food, Beverage & Dining (30 Signs)
`apple`, `banana`, `bread`, `breakfast`, `butter`, `cake`, `cheese`, `coffee`, `cook`, `dinner`, `drink`, `eat`, `egg`, `fish`, `food`, `fruit`, `hungry`, `lunch`, `meat`, `milk`, `onion`, `orange`, `potato`, `rice`, `salt`, `sugar`, `tea`, `thirsty`, `vegetable`, `water`

### 4. Transport, Travel & Public Facilities (26 Signs)
`airport`, `ambulance`, `auto`, `bag`, `bicycle`, `boat`, `bridge`, `bus`, `car`, `city`, `crossroad`, `direction`, `flight`, `hotel`, `left`, `map`, `metro`, `platform`, `railway`, `right`, `road`, `station`, `ticket`, `traffic`, `train`, `vehicle`

### 5. Education, Work & Technology (32 Signs)
`book`, `calculator`, `calendar`, `camera`, `cellphone`, `class`, `college`, `computer`, `desk`, `dictionary`, `exam`, `internet`, `laptop`, `learn`, `lesson`, `library`, `money`, `office`, `paper`, `pen`, `pencil`, `phone`, `read`, `rupees`, `salary`, `school`, `student`, `study`, `teacher`, `telephone`, `university`, `write`

### 6. Nature, Animals & Calendar (35 Signs)
`afternoon`, `animal`, `bird`, `cat`, `cloud`, `cold`, `cow`, `day`, `dog`, `donkey`, `earth`, `elephant`, `fire`, `flower`, `forest`, `horse`, `hot`, `lion`, `monkey`, `month`, `moon`, `morning`, `mountain`, `night`, `rain`, `river`, `season`, `sky`, `snake`, `sun`, `tiger`, `tree`, `weather`, `wind`, `year`

### 7. Household, Clothing & Objects (40 Signs)
`bag`, `ball`, `bed`, `bedroom`, `blanket`, `bottle`, `box`, `bucket`, `chair`, `clock`, `clothes`, `coat`, `comb`, `cup`, `curtain`, `door`, `dress`, `fan`, `furniture`, `glass`, `hat`, `house`, `key`, `knife`, `lamp`, `lock`, `mirror`, `needle`, `pillow`, `plate`, `room`, `rope`, `scissor`, `shirt`, `shoe`, `soap`, `sofa`, `table`, `towel`, `window`

### 8. Descriptors, Colors & Common Actions (38 Signs)
`alright`, `bad`, `beautiful`, `biglarge`, `black`, `blue`, `bright`, `brown`, `cheap`, `clean`, `close`, `dark`, `dirty`, `easy`, `expensive`, `fast`, `fine`, `finish`, `green`, `happy`, `hard`, `heavy`, `light`, `loud`, `new`, `old`, `open`, `pink`, `quiet`, `ready`, `red`, `sad`, `slow`, `small`, `strong`, `weak`, `white`, `yellow`

---

## ⚡ Execution Modality

The AI4Bharat INCLUDE model operates in two execution modes:
1. **Isolated Video Mode (`/api/v1/predict/video`):** Upload an MP4 video clip; the service extracts 134-D spatio-temporal keypoints, performs temporal sequence interpolation (32 frames), and outputs the top predicted sign.
2. **Real-Time Streaming Adapter (`StreamingIncludeAdapter`):** Consumes streaming frame coordinates in a sliding 32-frame buffer with confidence hysteresis to emit recognized signs continuously during live webcam feeds.
