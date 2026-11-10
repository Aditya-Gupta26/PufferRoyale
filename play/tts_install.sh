#!/bin/bash
# Install the neural voice for the play site's commentary (run by the owner; Claude's auto mode may not install
# outside code itself). Chatterbox TTS 0.1.7 (Resemble AI, MIT licence): a free, open-source speech model with an
# "exaggeration" control for expressive, human-sounding delivery. Nothing here costs money.
#
#     bash /scratch/$USER/pufferroyale/PufferRoyale/play/tts_install.sh
#
# What it does:
#   1. pip-installs chatterbox-tts 0.1.7 and its 73 Python dependencies (exact versions, resolved 2026-11-06) into
#      $PR_ROOT/tools/tts_py, with --no-deps: it does NOT install torch (the image already has torch 2.6.0, the
#      version Chatterbox requires), pins torchaudio to 2.6.0 (the resolver had picked 2.11, which needs torch 2.11),
#      and leaves out gradio (only Chatterbox's demo web page). setuptools 80.9 is included for pkg_resources
#      (the watermarker needs it; the image's venv has no setuptools; setuptools >= 81 dropped pkg_resources).
#      The container image itself is not changed.
#   2. downloads the model weights from Hugging Face (ResembleAI/chatterbox, about 3 GB) into
#      $PR_ROOT/tools/hf_cache.
# Undo: rm -rf $PR_ROOT/tools/tts_py $PR_ROOT/tools/hf_cache
set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
T="$PR_ROOT/tools/tts_py"
H="$PR_ROOT/tools/hf_cache"
mkdir -p "$T" "$H"
REQS=$(mktemp)
cat > "$REQS" <<'EOF'
annotated-doc==0.0.5
anyio==4.16.0
audioread==3.1.0
catalogue==2.0.10
certifi==2026.10.23
cffi==2.1.1
cfgv==3.5.0
charset-normalizer==3.5.2
click==8.5.1
cloudpickle==3.1.2
conformer==0.3.2
decorator==5.3.1
deprecated==3.0.1
diffusers==0.29.0
distlib==0.4.4
einops==0.8.2
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.33.0
identify==2.6.20
idna==3.20
importlib_metadata==10.1.0
jaconv==0.5.0
joblib==1.6.0
lazy-loader==0.6
librosa==0.11.0
llvmlite==0.50.0
markdown-it-py==4.2.0
mdurl==0.1.2
ml_dtypes==0.5.4
msgpack==1.2.4
narwhals==2.28.0
nodeenv==1.12.0
numba==0.68.1
omegaconf==2.4.0
onnx==1.23.2
packaging==26.3
pillow==13.0.0
platformdirs==4.14.5
pooch==1.9.0
pre_commit==4.7.0
protobuf==7.36.2
pycparser==3.12
pygments==2.21.0
pykakasi==2.3.0
pyloudnorm==0.2.0
python-discovery==1.9.5
pyyaml==6.0.3
regex==2026.11.7
requests==2.34.2
resemble-perth==1.0.1
rich==15.0.0
s3tokenizer==0.3.0
safetensors==0.5.3
scikit-learn==1.9.1
shellingham==1.5.4
soundfile==0.14.0
soxr==1.1.0
spacy_pkuseg==1.0.2
srsly==2.5.6
threadpoolctl==3.8.0
tokenizers==0.22.2
tqdm==4.70.1
transformers==5.2.0
typer-slim==0.24.0
typer==0.28.0
urllib3==2.8.0
virtualenv==21.15.6
wrapt==2.5.2
zipp==4.1.2
torchaudio==2.6.0
setuptools==80.9.0
chatterbox-tts==0.1.7
EOF
echo "[1/2] installing $(wc -l < "$REQS") packages into $T (no torch)"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" "$SIF" python3 -m pip install --quiet --no-deps --target "$T" -r "$REQS"
rm -f "$REQS"
echo "[2/2] downloading the Chatterbox weights into $H (the model's own loader fetches exactly the files it needs)"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --env "PYTHONPATH=$T,HF_HOME=$H,OMP_NUM_THREADS=4" "$SIF" python3 -c "
import torch; torch.set_num_threads(4)
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device='cpu')
print('model loaded; sample rate', m.sr)"
du -sh "$T" "$H"
echo "done: tell Claude the voice is installed"
