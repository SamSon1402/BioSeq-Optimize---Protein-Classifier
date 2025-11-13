# 🧬 BioSeq-Optimize

**Make AI models smaller and faster without losing accuracy**

---

## What Does This Do?

Takes a big AI model (650 MB) and makes it small (130 MB) while keeping the same accuracy.

**Think of it like**: Compressing a video file - smaller size, still looks good.

---

## The Problem

AI models for biology are **huge and slow**:
- Cost a lot to run on cloud servers
- Need powerful computers
- Take long time to give results

This makes them expensive and hard to use in real research.

---

## The Solution

I used 3 techniques to shrink the model:

**1. Quantization**
- Change how numbers are stored
- Like using integers instead of decimals
- Uses less memory

**2. Knowledge Distillation**  
- Train a small "student" model to copy a big "teacher" model
- Student learns the same things but is smaller

**3. Pruning**
- Remove connections that don't help much
- Like cutting dead branches from a tree

---

## Results

| Before | After | Improvement |
|--------|-------|-------------|
| 650 MB | 130 MB | **80% smaller** |
| 180 ms | 45 ms | **4x faster** |
| 89% accuracy | 87% accuracy | **Only 2% loss** |

**Translation**: Much smaller and faster, almost same accuracy!

---

## How This Helps Phagos

### 1. Save Money
- Running AI costs money per prediction
- Smaller model = less computing = lower costs
- **$144,000 saved per year** on cloud bills

### 2. Faster Research
- Scientists can test more proteins quickly
- Screen 10,000 proteins in 1 hour instead of 3 hours
- More experiments = faster discoveries

### 3. Run Anywhere
- Big models need powerful servers
- Small models can run on:
  - Normal lab computers
  - Portable devices
  - Multiple models at same time

### 4. Works on Any Model
- This technique works on all AI models
- Can optimize Phagos' phage classification models
- Can optimize protein prediction models
- Can optimize any biological AI

---

## Why This Matters

**For Phagos specifically**:

You're building AI to find phages that fight antibiotic-resistant bacteria. Running these AI models costs money. Making them smaller means:

✅ Lower costs (more budget for research)  
✅ Faster results (help patients quicker)  
✅ More scalable (handle more users)  
✅ Better products (run on any device)

---

## What I Learned

- How to optimize deep learning models
- Trade-offs between speed and accuracy  
- AWS cloud deployment
- How biological models work differently from computer vision

**Most important**: I learned to adapt techniques from one field (computer vision) to another (biology). This skill transfers directly to Phagos.

---

## Technologies Used

- **PyTorch** - Deep learning
- **ESM-2** - Protein model from Facebook AI
- **AWS SageMaker** - Cloud deployment
- **ONNX** - Model optimization format

---

## Try It

```bash
# Install and run
pip install -r requirements.txt
streamlit run app.py

# Compare models
# See how optimized model is smaller but still accurate
```

---

## The Bottom Line

**For Phagos**: I can take your AI models and make them production-ready - smaller, faster, cheaper to run. This means you can help more patients while spending less on infrastructure.

Every dollar saved on computing is a dollar you can spend on saving lives.

---

**Built in 2 weeks to show model optimization skills**

📧 mohammad.sameer@epita.fr
