# การใช้งาน venv และการ activate ด้วย uv

## สร้าง venv

```bash
uv venv                # สร้าง virtual environment (โฟลเดอร์ .venv)
uv venv --python 3.12  # สร้าง venv ด้วย Python เวอร์ชันที่ระบุ
```

## ติดตั้งแพ็กเกจ

- สำหรับโปรเจกต์ที่มี pyproject.toml
	```bash
	uv add requests
	```
- สำหรับ venv เปล่า (ไม่มี pyproject.toml)
	```bash
	uv pip install requests
	```

## การรันคำสั่งใน venv โดยไม่ต้อง activate

```bash
uv run python -V
uv run python app.py
```

## การ activate venv (ตัวเลือก)

```bash
source .venv/bin/activate   # zsh/bash (macOS)
deactivate                   # ออกจาก venv
# fish: source .venv/bin/activate.fish
# csh/tcsh: source .venv/bin/activate.csh
```

> **Tip:** เมื่อใช้ uv run หรือ uv pip ไม่จำเป็นต้อง activate venv
# คู่มือใช้งาน uv เบื้องต้น (Quickstart)

เอกสารนี้สรุปขั้นตอนติดตั้งและใช้งานพื้นฐานของ uv ตามแนวทางจากเว็บไซต์ทางการ https://docs.astral.sh/uv/ โดยเรียบเรียงเป็นภาษาไทยแบบย่อเพื่อเริ่มต้นใช้งานได้ทันที (โปรดอ้างอิงเอกสารทางการสำหรับรายละเอียดและตัวเลือกทั้งหมด)

หมายเหตุ: uv คือเครื่องมือจัดการแพ็กเกจและโปรเจกต์ Python ที่รวดเร็วมาก รองรับการทำงานคล้าย pip/pip-tools/pipx/poetry/pyenv/virtualenv ฯลฯ ในเครื่องมือเดียว


## ติดตั้ง uv

macOS/Linux (แนะนำตามทางการ):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

หลังติดตั้ง เสนอให้ดูหน้า “First steps” และคู่มือการติดตั้งฉบับเต็มบนเว็บไซต์ทางการ หากต้องการติดตั้งด้วยช่องทางอื่น (เช่น pip หรือ Homebrew) ดูที่ Installation page บน docs


## คำสั่งพื้นฐานที่ควรรู้

- uv init: สร้างโปรเจกต์ใหม่ (pyproject.toml)
- uv add / uv remove: เพิ่ม/ลบ dependencies ของโปรเจกต์
- uv run: รันคำสั่งหรือสคริปต์ภายใต้สภาพแวดล้อมของโปรเจกต์
- uv lock / uv sync: อัปเดต lockfile และซิงก์ environment ให้ตรงตาม lockfile
- uv venv: สร้าง virtual environment
- uv tool / uvx: ติดตั้ง/รัน CLI tools จากแพ็กเกจ Python
- uv python: จัดการเวอร์ชัน Python (ติดตั้ง/ค้นหา/ปักหมุดเวอร์ชัน)
- uv pip: อินเทอร์เฟซที่เข้ากันได้กับ pip/pip-tools (compile/sync/install/uninstall ฯลฯ)


## เริ่มโปรเจกต์ใหม่และติดตั้งแพ็กเกจ

```bash
# 1) สร้างโปรเจกต์
uv init myapp
cd myapp

# 2) เพิ่มแพ็กเกจ (ตัวอย่าง: ruff)
uv add ruff

# 3) รันคำสั่งโดยใช้ environment ของโปรเจกต์
uv run ruff --version

# 4) อัปเดต lockfile และซิงก์ environment
uv lock
uv sync
```

เคล็ดลับ:
- uv จะสร้าง/ใช้ .venv ให้โดยอัตโนมัติ (ไม่จำเป็นต้อง activate เองเมื่อใช้ uv run/uv sync ฯลฯ)
- หากต้องการดูโครงสร้าง dependencies ใช้ `uv tree`


## จัดการสคริปต์เดี่ยว (PEP 723 Inline metadata)

```bash
# สร้างไฟล์สคริปต์อย่างง่าย
echo 'import requests; print("ok", bool(requests))' > script.py

# เพิ่ม dependency ให้กับสคริปต์
uv add --script script.py requests

# รันสคริปต์ใน environment ชั่วคราวที่จัดการให้
uv run script.py
```


## ใช้งานเครื่องมือ (CLI tools) ผ่าน uvx/uv tool

- รันเครื่องมือแบบไม่ติดตั้งถาวร (ephemeral):

```bash
uvx pycowsay 'hello world'
```

- ติดตั้งเครื่องมือสำหรับใช้งานซ้ำ:

```bash
uv tool install ruff
ruff --version
```


## จัดการ Python เวอร์ชัน

```bash
# ติดตั้ง Python หลายเวอร์ชัน
uv python install 3.10 3.11 3.12

# สร้าง venv ระบุเวอร์ชัน Python
uv venv --python 3.12.0

# ปักหมุดเวอร์ชันที่ใช้ในไดเรกทอรีปัจจุบัน
uv python pin 3.11
```


## อินเทอร์เฟซสไตล์ pip (เร่งความเร็ว workflow เดิมของคุณ)

- Compile requirements แบบข้ามแพลตฟอร์ม (ทำนอง pip-tools):

```bash
uv pip compile requirements.in --universal -o requirements.txt
```

- สร้าง venv และติดตั้งแพ็กเกจตามไฟล์ที่ล็อกไว้:

```bash
uv venv
uv pip sync requirements.txt
```

- ติดตั้ง/ถอดแพ็กเกจลง venv ปัจจุบัน:

```bash
uv pip install requests
uv pip uninstall requests
```


## ขั้นตอนแนะนำสำหรับโครงการจริง (สรุปสั้น)

1) uv init เพื่อสร้างโปรเจกต์ และ commit pyproject.toml
2) uv add เพื่อติดตั้ง dependencies แรกเริ่ม (เช่น ruff, pytest ฯลฯ)
3) uv lock เพื่อสร้าง/อัปเดต uv.lock และ uv sync เพื่อให้ environment ตรงกับ lockfile
4) ใช้ uv run ในการรันคำสั่งพัฒนา/ทดสอบ (เช่น uv run pytest)
5) ใช้ uv tool/uvx เมื่อต้องการ CLI tools เพิ่มเติม (เช่น ruff, black)
6) หากต้องการควบคุม Python เวอร์ชัน ใช้ uv python install/pin และระบุ --python เมื่อจำเป็น


## ทิปส์และการแก้ปัญหาเบื้องต้น

- โหมดออฟไลน์: เพิ่ม --offline เมื่อเครือข่ายจำกัด
- เร่งการติดตั้งซ้ำ: ใช้ --reinstall หรือ --refresh เมื่อแคชเก่าไม่ตรง
- ป้องกันแพ็กเกจส่วนเกิน: ใช้ `uv sync` (แบบ exact โดยดีฟอลต์) หรือ `uv pip sync --exact`
- ดูตำแหน่งแคช: `uv cache dir` และล้าง/ปรับแต่งด้วยคำสั่งในกลุ่ม `uv cache`


## แหล่งอ้างอิงทางการ

- หน้าแรก uv: https://docs.astral.sh/uv/
- Getting started: https://docs.astral.sh/uv/getting-started/
- Commands (CLI Reference): https://docs.astral.sh/uv/reference/cli/
- pip interface: https://docs.astral.sh/uv/pip/
- Guides: https://docs.astral.sh/uv/guides/

(ข้อมูลทั้งหมดอาจมีการเปลี่ยนแปลงตามเวอร์ชันของ uv โปรดตรวจสอบเอกสารทางการเสมอ)
