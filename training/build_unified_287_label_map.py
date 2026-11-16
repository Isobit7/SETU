"""
SETU — Build Unified 287-Class Dynamic Vocabulary
Merges 53 dynamic classes (Amrita phrases + greetings + no_sign)
and 263 AI4Bharat INCLUDE vocabulary classes, deduplicating 29 overlapping signs.
Output: models/metadata/dynamic_unified_287_label_map.json
"""

import json
from pathlib import Path

def normalize_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

def format_clean_display(label: str) -> str:
    # Handle known multi-word phrases and common terms
    display_overrides = {
        "no_sign": "Waiting for a sign...",
        "hello": "HELLO",
        "thank_you": "THANK YOU",
        "thankyou": "THANK YOU",
        "good_morning": "GOOD MORNING",
        "goodmorning": "GOOD MORNING",
        "good_afternoon": "GOOD AFTERNOON",
        "goodafternoon": "GOOD AFTERNOON",
        "good_evening": "GOOD EVENING",
        "goodevening": "GOOD EVENING",
        "good_night": "GOOD NIGHT",
        "goodnight": "GOOD NIGHT",
        "how_are_you": "HOW ARE YOU?",
        "howareyou": "HOW ARE YOU?",
        "i": "I (ME)",
        "you": "YOU",
        "youplural": "YOU (PLURAL)",
        "namaste": "NAMASTE",
        "are_you_free_today": "Are you free today?",
        "be_careful": "Be careful",
        "clean_the_room": "Clean the room",
        "close_the_door": "Close the door",
        "come_here": "Come here",
        "dont_worry": "Don't worry",
        "drive_slowly": "Drive slowly",
        "give_me_your_pen": "Give me your pen",
        "go_to_market": "Go to the market",
        "have_you_taken_medicine": "Have you taken medicine?",
        "how_much_does_this_cost": "How much does this cost?",
        "how_old_are_you": "How old are you?",
        "i_am_fine": "I am fine",
        "i_am_sorry": "I am sorry",
        "i_love_you": "I love you",
        "i_miss_you": "I miss you",
        "lets_go_out": "Let's go out",
        "nice_to_meet_you": "Nice to meet you",
        "open_the_window": "Open the window",
        "please_sit_down": "Please sit down",
        "see_you_later": "See you later",
        "stand_up": "Stand up",
        "switch_off_the_light": "Switch off the light",
        "take_rest": "Take rest",
        "what_are_you_doing": "What are you doing?",
        "what_is_the_time": "What is the time?",
        "what_is_your_name": "What is your name?",
        "where_do_you_live": "Where do you live?",
        "where_is_the_hospital": "Where is the hospital?",
        "biglarge": "BIG / LARGE",
        "smalllittle": "SMALL / LITTLE",
        "storeorshop": "STORE / SHOP",
        "streetorroad": "STREET / ROAD",
        "raceethnicity": "RACE / ETHNICITY",
        "secondnumber": "SECOND (NUMBER)",
        "trainstation": "TRAIN STATION",
        "trainticket": "TRAIN TICKET",
        "tshirt": "T-SHIRT",
    }
    if label in display_overrides:
        return display_overrides[label]
    # Default uppercase with spaces
    return label.replace("_", " ").upper()

def main():
    root = Path(__file__).resolve().parent.parent
    master_path = root / "models" / "metadata" / "dynamic_master_label_map.json"
    include_path = root / "models" / "pretrained" / "label_map_include.json"

    with open(master_path, "r", encoding="utf-8") as f:
        master_data = json.load(f)
        master_classes = master_data.get("classes", [])

    with open(include_path, "r", encoding="utf-8") as f:
        include_dict = json.load(f)
        include_classes = list(include_dict.keys())

    print(f"Loaded {len(master_classes)} master dynamic classes")
    print(f"Loaded {len(include_classes)} INCLUDE classes")

    # Map normalized key -> canonical label
    # Master dynamic classes take preference for canonical name formatting
    # (e.g. good_morning instead of goodmorning, thank_you instead of thankyou)
    unified_entries = {}

    for c in master_classes:
        norm = normalize_key(c)
        unified_entries[norm] = {
            "canonical_label": c,
            "display_name": format_clean_display(c),
            "sources": ["amrita_dynamic"],
            "master_class": c,
            "include_class": None,
        }

    duplicate_count = 0
    for inc in include_classes:
        norm = normalize_key(inc)
        if norm in unified_entries:
            # Duplicate sign detected! Merge into existing entry
            duplicate_count += 1
            unified_entries[norm]["sources"].append("include_263")
            unified_entries[norm]["include_class"] = inc
        else:
            unified_entries[norm] = {
                "canonical_label": inc,
                "display_name": format_clean_display(inc),
                "sources": ["include_263"],
                "master_class": None,
                "include_class": inc,
            }

    print(f"Found and deduplicated {duplicate_count} duplicate signs across both sets.")

    # Sort canonical labels alphabetically, keeping no_sign at index 0 or at the end
    # Having no_sign at index 0 is conventional for idle/rejection class
    sorted_norms = sorted(unified_entries.keys())
    # Ensure no_sign is handled cleanly
    no_sign_norm = normalize_key("no_sign")
    if no_sign_norm in sorted_norms:
        sorted_norms.remove(no_sign_norm)
        sorted_norms = [no_sign_norm] + sorted_norms

    classes_list = []
    display_names = {}
    label_to_index = {}
    class_details = {}

    for idx, norm in enumerate(sorted_norms):
        info = unified_entries[norm]
        label = info["canonical_label"]
        classes_list.append(label)
        label_to_index[label] = idx
        display_names[label] = info["display_name"]
        class_details[label] = {
            "index": idx,
            "display_name": info["display_name"],
            "sources": info["sources"],
            "master_class": info["master_class"],
            "include_class": info["include_class"],
        }

    total_classes = len(classes_list)
    print(f"Final deduplicated unified vocabulary count: {total_classes} classes")

    output_payload = {
        "model_name": "SETU-Unified-Dynamic-287-ISL",
        "total_classes": total_classes,
        "classes": classes_list,
        "label_to_index": label_to_index,
        "display_names": display_names,
        "class_details": class_details,
        "duplicate_signs_count": duplicate_count,
        "sources": {
            "master_dynamic_count": len(master_classes),
            "include_vocabulary_count": len(include_classes),
            "overlapping_deduplicated_count": duplicate_count,
        },
    }

    out_file = root / "models" / "metadata" / "dynamic_unified_287_label_map.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved unified label map to: {out_file}")

if __name__ == "__main__":
    main()

