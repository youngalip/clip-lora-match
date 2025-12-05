# scripts/rebuild_index.py
"""
Script untuk rebuild index dari database.
Jalankan ini jika index .pt tidak sinkron dengan database.

Usage:
    python scripts/rebuild_index.py
"""

from pathlib import Path
import torch
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.db.database import SessionLocal
from src.db.models import FoundItem
from models.clip_model import load_clip_model, encode_text

# Config paths
CLIP_CONFIG = ROOT_DIR / "config" / "clip_config.yaml"
LORA_DIR = ROOT_DIR / "models" / "saved" / "clip-lora" / "epoch_1"
INDEX_PATH = ROOT_DIR / "data" / "index" / "custom_items_index.pt"


def rebuild_index():
    """
    Rebuild index dari semua item di database.
    """
    print("=" * 60)
    print("REBUILDING INDEX FROM DATABASE")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1/4] Loading CLIP model...")
    model, processor, device = load_clip_model(
        config_path=CLIP_CONFIG,
        use_lora=True,
        lora_weights_path=LORA_DIR,
    )
    print(f"✓ Model loaded on {device}")
    
    # 2. Get all items from database
    print("\n[2/4] Fetching items from database...")
    db = SessionLocal()
    try:
        items = db.query(FoundItem).order_by(FoundItem.id).all()
        print(f"✓ Found {len(items)} items in database")
    finally:
        db.close()
    
    if not items:
        print("\n⚠️  No items in database. Nothing to rebuild.")
        return
    
    # 3. Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embeddings_list = []
    image_paths = []
    texts = []
    
    for i, item in enumerate(items, 1):
        print(f"  Processing item {i}/{len(items)}: {item.id}")
        
        # Generate embedding from text
        text_emb = encode_text(item.description, model, processor, device)
        
        # Ensure shape is (1, D)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        
        # Normalize
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        embeddings_list.append(text_emb)
        image_paths.append(item.image_path)
        texts.append(item.description)
    
    # Stack all embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)
    print(f"✓ Generated {all_embeddings.shape[0]} embeddings")
    
    # 4. Save index
    print("\n[4/4] Saving index...")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "embeddings": all_embeddings,
            "image_paths": image_paths,
            "texts": texts,
        },
        INDEX_PATH,
    )
    
    print(f"✓ Index saved to: {INDEX_PATH}")
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    data = torch.load(INDEX_PATH)
    print(f"Embeddings shape: {data['embeddings'].shape}")
    print(f"Image paths: {len(data['image_paths'])} items")
    print(f"Texts: {len(data['texts'])} items")
    
    if (data['embeddings'].shape[0] == len(data['image_paths']) == len(data['texts'])):
        print("\n✅ Index is synchronized!")
    else:
        print("\n❌ Index is NOT synchronized!")
    
    print("\nRebuild complete!")


if __name__ == "__main__":
    try:
        rebuild_index()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)