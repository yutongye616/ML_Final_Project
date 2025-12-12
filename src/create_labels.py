import os
import xml.etree.ElementTree as ET
import pandas as pd

VOC_ROOT = os.path.join("data", "VOC2012")
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")

CLASSES = ["person", "car", "dog", "cat", "bicycle"]

def main():
    rows = []

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} annotation files")

    for file in xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        labels = {cls: 0 for cls in CLASSES}

        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if name in labels:
                labels[name] = 1

        image_id = file.replace(".xml", "")
        row = {"image_id": image_id}
        row.update(labels)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "labels_5classes.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
