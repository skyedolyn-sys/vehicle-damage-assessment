import os
import requests
import json
import time

folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
files = []
for f in sorted(os.listdir(folder)):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        files.append(("files", open(os.path.join(folder, f), "rb")))

print(f"Uploading {len(files)} photos...")

upload_url = "http://localhost:8000/api/upload"
resp = requests.post(upload_url, files=files, data={
    "brand": "蔚来",
    "model": "ES8",
    "year": "2019"
})
print(f"Upload status: {resp.status_code}")
upload_data = resp.json()
print(f"Task ID: {upload_data['task_id']}")

task_id = upload_data["task_id"]
assess_url = f"http://localhost:8000/api/assess/{task_id}?brand=%E8%94%9A%E6%9D%A5&model=ES8&year=2019"

print("\nStarting SSE stream...")
resp = requests.get(assess_url, stream=True)
final_result = None
for line in resp.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("event:"):
            event = line.replace("event: ", "").strip()
        elif line.startswith("data:"):
            data = json.loads(line.replace("data: ", ""))
            if event == "step":
                print(f"[Step {data.get('step')}] {data.get('message')}")
            elif event == "complete":
                print("\n✅ Assessment complete")
            elif event == "error":
                print(f"\n❌ Error: {data.get('message')}")
            elif event == "result":
                final_result = data
                print("\n📋 Result received")

if final_result:
    with open("/Users/sky/Downloads/vehicle_damage_assessment/result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print("\nSaved to result.json")
    print(f"Overall severity: {final_result['assessment_summary']['overall_severity']}")
    print(f"Structural damage: {final_result['assessment_summary']['structural_damage_flag']}")
    print(f"Damaged parts: {final_result['assessment_summary']['damaged_parts_count']}")
    print(f"Uncertain parts: {final_result['assessment_summary']['uncertain_parts_count']}")
