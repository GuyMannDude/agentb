#!/bin/bash
# ─────────────────────────────────────────────
# AgentB — One-Command Install
# https://github.com/GuyMannDude/agentb
# ─────────────────────────────────────────────
set -e

echo "🧠 AgentB — Installing..."
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 required. Install it first."
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VER"

# Install directory
INSTALL_DIR="${AGENTB_DIR:-$HOME/.local/share/agentb}"
DATA_DIR="${AGENTB_DATA:-$HOME/.agentb}"

echo "  Install dir: $INSTALL_DIR"
echo "  Data dir: $DATA_DIR"
echo ""

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "[1/4] Updating..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "[1/4] Cloning..."
    git clone --quiet https://github.com/GuyMannDude/agentb.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Virtual environment
echo "[2/4] Setting up environment..."
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Config
echo "[3/4] Configuring..."
mkdir -p "$DATA_DIR"/{memory,cache/l1,cache/l2,logs}

CONFIG_FILE="$HOME/.config/agentb/agentb.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cp agentb.yaml.example "$CONFIG_FILE"
    echo "  Created config: $CONFIG_FILE"
    echo "  ⚠️  Edit this file to set your provider and API keys!"
else
    echo "  Config exists: $CONFIG_FILE"
fi

# Create launcher script
echo "[4/4] Creating launcher..."
LAUNCHER="$HOME/.local/bin/agentb"
mkdir -p "$(dirname "$LAUNCHER")"
cat > "$LAUNCHER" << SCRIPT
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
exec python -m agentb.server "\$@"
SCRIPT
chmod +x "$LAUNCHER"

echo ""
echo "═══════════════════════════════════════════"
echo "🧠 AgentB installed!"
echo ""
echo "  Config:  $CONFIG_FILE"
echo "  Start:   agentb"
echo "  Health:  curl http://localhost:50001/health"
echo ""
echo "  Quick test:"
echo "    agentb &"
echo "    curl -X POST http://localhost:50001/context \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"prompt\": \"test query\"}'"
echo ""
echo "  Docker:"
echo "    cd $INSTALL_DIR"
echo "    docker compose up -d"
echo "═══════════════════════════════════════════"
