# GPU Training Solution for Barkour

## ⚠️ THE PROBLEM

**JAX does NOT support GPU on Windows.** This is a fundamental limitation:

```
Your System:
- OS: Windows 10/11 ✓
- GPU: NVIDIA RTX 2060 ✓
- CUDA: 12.9 and 13.0 installed ✓
- JAX: 0.8.0 installed ✓

Result: JAX devices = [CpuDevice(id=0)]  ❌
```

**Why?** JAX on Windows is CPU-only by design. Even with CUDA installed, JAX Windows builds don't include GPU support.

## ✅ THE SOLUTION

The **original tutorial.ipynb** was designed for **Google Colab**, not Windows. Here's what you need to do:

### Option 1: Google Colab (Recommended - What the Tutorial Uses)

This is exactly how the original tutorial is meant to be used:

1. **Upload Notebook**
   - Go to https://colab.research.google.com/
   - Upload `train_barkour_colab.ipynb`

2. **Enable GPU**
   - Click: Runtime → Change runtime type
   - Select: **T4 GPU** (free) or **A100** (paid)
   - Click: Save

3. **Run Training**
   - Click: Runtime → Run all (Ctrl+F9)
   - Wait ~30-45 minutes on T4 GPU
   - Wait ~6-10 minutes on A100 GPU

4. **Download Model**
   - After training completes
   - Click Files icon (📁) in left sidebar
   - Navigate to `/tmp/mjx_brax_quadruped_policy`
   - Right-click → Download

5. **Use Locally**
   - Transfer downloaded model to Windows
   - Place in `c:\users\hatem\Desktop\MuJoCo\trained_barkour_policy`
   - Run inference with `run_barkour_local.py`

### Option 2: WSL2 + CUDA (Advanced - NOT Recommended)

This is complex and may not work:

1. Install WSL2 (Windows Subsystem for Linux)
2. Install Ubuntu 22.04 in WSL2
3. Install NVIDIA drivers for WSL2
4. Install CUDA toolkit in WSL2
5. Install JAX with CUDA in WSL2
6. Run training in WSL2

**Problems:**
- Complex setup (1-2 hours)
- May not work with your GPU
- Requires 20GB+ disk space
- Performance might be worse than Colab

### Option 3: CPU Training (Slow but Works)

Keep training locally but accept slow speeds:

```bash
python train_barkour_local.py --no-prompt
```

**Performance:**
- 100K steps: ~10 minutes (CPU) vs. ~30 seconds (GPU)
- 1M steps: ~1 hour (CPU) vs. ~5 minutes (GPU)
- 5M steps: ~5 hours (CPU) vs. ~30 minutes (GPU)
- 100M steps: ~100 hours (CPU) vs. ~6 minutes (GPU A100)

## 📊 Comparison Table

| Method | GPU Support | Setup Time | Training Speed | Cost |
|--------|-------------|------------|----------------|------|
| **Google Colab T4** | ✅ Yes | 2 minutes | Fast (30-45 min for 100M) | Free |
| **Google Colab A100** | ✅ Yes | 2 minutes | Very Fast (6-10 min for 100M) | ~$1/hour |
| **Windows Local** | ❌ No | 0 minutes | Very Slow (100+ hours for 100M) | Free |
| **WSL2** | ⚠️ Maybe | 1-2 hours | Medium (if works) | Free |

## 🎯 Recommended Workflow

### For Training:
1. Use **Google Colab** with T4 GPU (free tier)
2. Upload `train_barkour_colab.ipynb`
3. Train for 100M steps (~30-45 minutes)
4. Download trained model

### For Inference/Testing:
1. Use **Windows locally** (works fine on CPU)
2. Use `run_barkour_local.py` 
3. Load the model trained on Colab
4. Test different locomotion scenarios

## 📝 Summary

**The original tutorial.ipynb is designed for Google Colab, not Windows.**

Key facts:
- ✅ Tutorial works perfectly on **Google Colab with GPU**
- ❌ Tutorial **cannot** run on Windows with GPU (JAX limitation)
- ✅ Inference works fine on **Windows CPU**
- ⚠️ Training on **Windows CPU** is 100x slower

**Bottom line:** Use Google Colab for training (as intended by the tutorial), then use Windows for inference.

## 🚀 Quick Start (Recommended Path)

```bash
# Step 1: Upload train_barkour_colab.ipynb to Google Colab
# Step 2: Select T4 GPU runtime
# Step 3: Run all cells (wait 30-45 min)
# Step 4: Download trained model from Colab

# Step 5: Test locally on Windows (CPU is fine for inference)
cd c:\users\hatem\Desktop\MuJoCo
python run_barkour_local.py
```

This matches exactly how the original tutorial is designed to work! 🎉
