<div align="center">
  <img src="assets/logo.png" alt="AnnotateANU Logo" width="200"/>

  # AnnotateANU

  ### Annotate at the Speed of AI. 100% Private.

  <p>
    AnnotateANU combines the power of Meta's SAM3 for instant segmentation with a strictly local-first architecture.<br/>
    Your images never leave your browser. Free, open-source, and built for high-performance computer vision workflows.
  </p>

  [![Open Source](https://img.shields.io/badge/Open%20Source-100%25-brightgreen)](https://github.com/agfianf/annotate-anu.git)
  [![Privacy](https://img.shields.io/badge/Privacy-No%20Server%20Uploads-blue)](https://github.com/agfianf/annotate-anu.git)
  [![Powered by SAM3](https://img.shields.io/badge/Powered%20by-Meta%20SAM3-0467DF)](https://huggingface.co/facebook/sam3)

  [Get Started](#quick-start) · [Report Bug](https://github.com/agfianf/annotate-anu/issues)

</div>

---

## 📚 Table of Contents

- [Why Choose AnnotateANU?](#why-choose-annotatanu)
- [Features](#-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Roadmap](#-roadmap---coming-soon)
- [Troubleshooting](#troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

- **⚡ Automated Segmentation**: SAM3 inference runs locally or via optimized endpoints to auto-segment objects instantly. Use text prompts or bounding boxes to get pixel-perfect masks in milliseconds.
  - **Text Prompts**: Describe objects in natural language ("person walking", "car on road")
  - **Bounding Box Exemplars**: Draw boxes around objects to find all similar instances
  - **Smart Prompt Memory**: Automatically remembers the last text prompt used for each label class

- **🔄 Intelligent Modes**: Choose the workflow that matches your task
  - **Single Mode**: Process one image at a time with full control
  - **Auto-Apply Mode**: Set your prompt once, automatically processes each new image
  - **Batch Mode**: Select multiple images and process them all with the same prompts

- **🎯 Manual Precision**: Need to tweak the AI's work? Use our pixel-perfect pen, rectangle, and polygon tools for fine-tuning your annotations with complete control.

- **📦 Batch Workflow**: Load hundreds of images at once. Our interface handles batch processing without browser lag, making large dataset annotation a breeze.

- **⌨️ Lightning Shortcuts**: Designed for power users. Keep your hands on the keyboard and annotate without breaking flow with comprehensive keyboard shortcuts.

- **💾 Export Ready**: Export to COCO JSON, YOLO format, or ZIP archives with one click. Industry-standard formats ready for your ML pipelines.

- **🔒 Local-First Storage**: Your data stays local with IndexedDB - no server uploads, total privacy. All processing happens in your browser or on your local backend.


![feature](assets/features.gif)


## Architecture

AnnotateANU is a simple monorepo with two independent applications and **two backend options**:

```
annotate-anu/                # Simple Monorepo
├── apps/
│   ├── web/                 # React annotation interface
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   ├── api-inference/       # FastAPI HuggingFace SAM3 backend (gated model)
│   │   ├── src/app/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── api-inference-yolo/  # FastAPI Ultralytics SAM3 backend (recommended)
│   │   ├── src/app/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── sam3.pt              # SAM3 model weights (not included, see setup)
├── venv/                    # Python virtual environment
├── docker-compose.yml       # Orchestrates all services
├── setup-yolo.sh/bat        # Setup script for Ultralytics backend
├── setup-hf.sh/bat          # Setup script for HuggingFace backend
├── run-yolo.sh/bat          # Run script for Ultralytics backend
├── run-hf.sh/bat            # Run script for HuggingFace backend
├── Makefile                 # Development commands
└── README.md
```

### Backend Options

**🚀 Ultralytics SAM3 (Recommended)** - `api-inference-yolo`
- ✅ Faster inference with FP16 support
- ✅ Better text prompt segmentation with semantic understanding
- ✅ Bounding box exemplar-based segmentation for finding similar objects
- ✅ No HuggingFace account required (but model download is manual)
- ✅ Supports single, auto-apply, and batch processing modes
- 📦 Uses: `ultralytics`, PyTorch, SAM3SemanticPredictor

**🔄 HuggingFace SAM3** - `api-inference`
- ✅ Auto-downloads model on first run
- ⚠️ Requires HuggingFace account and gated model access
- 📦 Uses: `transformers`, `huggingface-hub`

## Quick Start

### Prerequisites

- **Python 3.12+** (required for local setup)
- **Node.js 18+** and **npm** (required for frontend)
- **Docker & Docker Compose** (optional, for containerized setup)

### Choose Your Backend

#### Option 1: Ultralytics SAM3 (Recommended) ⚡

**Requirements:**
- Download `sam3.pt` model weights manually (see instructions below)
- No HuggingFace account needed

**Setup & Run:**

```bash
# 1. Clone the repository
git clone https://github.com/agfianf/annotate-anu.git
cd annotate-anu

# 2. Download SAM3 model weights
# Visit: https://huggingface.co/facebook/sam3
# Request access, then download sam3.pt
# Place it in: apps/sam3.pt

# 3. Run setup script
./setup-yolo.sh   # Linux/Mac
setup-yolo.bat    # Windows

# 4. Start the application
./run-yolo.sh     # Linux/Mac
run-yolo.bat      # Windows

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Option 2: HuggingFace SAM3

**Requirements:**
- HuggingFace account with gated model access
- Model auto-downloads on first run (~2.4GB)

**Setup & Run:**

```bash
# 1. Clone the repository
git clone https://github.com/agfianf/annotate-anu.git
cd annotate-anu

# 2. Setup HuggingFace token
# - Create account: https://huggingface.co/join
# - Request access: https://huggingface.co/facebook/sam3
# - Generate token: https://huggingface.co/settings/tokens

# 3. Run setup script
./setup-hf.sh     # Linux/Mac
setup-hf.bat      # Windows

# 4. Add your HuggingFace token
# Edit apps/api-inference/.env:
# HF_TOKEN=hf_your_token_here

# 5. Start the application
./run-hf.sh       # Linux/Mac
run-hf.bat        # Windows
```

### SAM3 Model Weights Setup

**⚠️ IMPORTANT: Unlike other Ultralytics models, SAM3 weights (`sam3.pt`) are NOT automatically downloaded.**

You must manually download the model weights:

1. **Request Access**: Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and click "Request Access"
2. **Wait for Approval**: You'll receive an email when approved (usually within a few hours)
3. **Download Model**: Once approved, go to the "Files" tab and download `sam3.pt` (~2.4GB)
4. **Place File**: Put `sam3.pt` in `apps/sam3.pt` (relative to project root)

```bash
# Correct location:
annotate-anu/
└── apps/
    └── sam3.pt    # Place the downloaded file here
```

### Docker Setup (Alternative)

```bash
# 1. Setup environment (choose your backend)
cp apps/api-inference-yolo/.env.example apps/api-inference-yolo/.env
# OR
cp apps/api-inference/.env.example apps/api-inference/.env

# 2. Add credentials and place sam3.pt in apps/

# 3. Start all services
make docker-up

# 4. Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Troubleshooting

### TypeError: 'SimpleTokenizer' object is not callable

If you encounter this error during prediction with Ultralytics SAM3:

```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat # Windows

# Fix the CLIP package conflict
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git
```

This error occurs when the wrong `clip` package is installed. The Ultralytics-specific CLIP package is required.

### Model Loading Issues

**Problem**: "SAM3 model weights not found"
- **Solution**: Ensure `sam3.pt` is in `apps/sam3.pt` directory
- Check file permissions: `ls -la apps/sam3.pt`

**Problem**: "CUDA out of memory"
- **Solution**: Reduce image size or switch to CPU mode in `.env`:
  ```bash
  SAM3_DEVICE=cpu
  ```

**Problem**: Backend shows "0 detections"
- **Solution**: Lower the confidence threshold (default 0.25)
- Try different text prompts (e.g., "object" instead of specific names)
- Check image quality and size

### Port Already in Use

```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9  # Kill backend
lsof -ti:5173 | xargs kill -9  # Kill frontend

# Windows
netstat -ano | findstr :8000  # Find PID
taskkill /F /PID <PID>        # Kill process
```

## 🚀 Roadmap - Coming Soon

We are constantly evolving. Here's what's shipping next to AnnotateANU:

#### 🎨 Enhanced Annotation Tools
- **Magic Wand Tool**: Click-to-segment for quick region selection
- **Edge Refinement**: AI-powered edge smoothing for precise mask boundaries
- **Annotation Templates**: Save and reuse common annotation patterns

#### 🔌 Bring Your Own Model (BYOM)
Connect your existing custom models via API. Pre-label your images using your own weights to bootstrap the annotation process even faster.

#### 🤖 Advanced AI Features
- **Active Learning**: Intelligently suggest which images to annotate next
- **Cross-Image Tracking**: Track objects across video frames or image sequences
- **Multi-Model Ensemble**: Combine predictions from multiple models for better accuracy

#### ☁️ Enterprise Storage Integration
Move beyond browser storage. We're adding native integration for MinIO and S3-compatible object storage, allowing you to pull and sync datasets directly from your cloud buckets.

#### 👥 Collaboration Features
- **Team Workspaces**: Share projects and annotations across team members
- **Review Mode**: Approve or reject annotations with comment threads
- **Version Control**: Track annotation history and changes over time

#### 📊 Analytics & Insights
- **Annotation Statistics**: Track productivity metrics and dataset balance
- **Quality Checks**: Automated validation for annotation consistency
- **Export Analytics**: Detailed reports on dataset composition


## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, we'd love your help.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Want to influence what we build next? Join our community on GitHub and share your ideas!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- [SAM3 Model (HuggingFace)](https://huggingface.co/facebook/sam3)
- [Ultralytics SAM3 Documentation](https://docs.ultralytics.com/models/sam-3/)
- [T-REX Label](https://www.trexlabel.com/)
- [MakeSense.ai](https://www.makesense.ai/)

## Acknowledgments

- **Meta AI** for the SAM3 (Segment Anything Model 3) architecture
- **Ultralytics** for the excellent SAM3 implementation and PyTorch optimization
- The open-source community for inspiration and tools


---

<div align="center">
  <p><strong>Ready to speed up your CV pipeline?</strong></p>
  <p>© 2025 AnnotateANU. Built for the Computer Vision Community.</p>
</div>

