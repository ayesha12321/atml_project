# Runpod Setup Guide

## 1. Rent GPU
- Go to runpod.io
- Choose: **Secure Cloud** or **Community Cloud**
- GPU: **A40** (48GB, ~$0.69/hr) or **RTX A6000** (48GB, ~$0.79/hr)
- Template: **PyTorch 2.1** or **RunPod PyTorch**
- Disk: 50GB minimum
- Click **Deploy**

## 2. Connect
- Once running, click **Connect** → **Start Jupyter Lab** or **SSH**
- Open terminal in Jupyter

## 3. Upload Files
**Option A: Direct upload in Jupyter**
- Click upload button, select all your files

**Option B: Via SSH/SCP**
```bash
# From your Windows PC (in PowerShell)
scp score_outputs.jsonl root@<pod-ip>:/workspace/
scp *.py root@<pod-ip>:/workspace/
scp requirements.txt root@<pod-ip>:/workspace/
```

## 4. Install Dependencies
```bash
cd /workspace
pip install -r requirements.txt
```

## 5. Run Experiment

**Step 1: Prepare data**
```bash
python prepare_data.py
```
Expected output: train_pairs.jsonl, val_pairs.jsonl
Time: ~2-5 minutes

**Step 2: Train verifier**
```bash
python train_verifier.py
```
Expected time: 2-4 hours
Checkpoint saved: verifier_best.pt

**Step 3: Score solutions**
```bash
python inference.py
```
Output: scored_outputs.jsonl

## 6. Download Results
**In Jupyter:**
- Right-click files → Download

**Via SCP:**
```bash
scp root@<pod-ip>:/workspace/verifier_best.pt ./
scp root@<pod-ip>:/workspace/scored_outputs.jsonl ./
```

## 7. Stop Pod
- Don't forget to **STOP** the pod when done to avoid charges!

## Monitoring Training
```bash
# Watch GPU usage
nvidia-smi -l 1

# Monitor training in real-time
tail -f nohup.out  # if running in background
```