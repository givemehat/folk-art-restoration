import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from models.registry import ModelRegistry
# Note: FolkArtInpaintDataset should be made modular to support both SR and Inpainting
from models.lama import FolkArtInpaintDataset 
from models.losses import ReconstructionLoss, PerceptualLoss, AdversarialLoss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_edsr_lama.yaml")
    parser.add_argument("--data_root", type=str, default="data/full")
    parser.add_argument("--save_dir", type=str, default="checkpoints/ablation")
    parser.add_argument("--mode", type=str, choices=["sr", "inpaint", "joint"], default="joint")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_cfg = config['training']
    
    # Init Registry
    registry = ModelRegistry(args.config)
    
    # Build models based on mode
    models = []
    if args.mode in ["sr", "joint"]:
        sr_model = registry.build_sr_model(device)
        models.append(sr_model)
    if args.mode in ["inpaint", "joint"]:
        inpaint_model = registry.build_inpaint_model(device)
        models.append(inpaint_model)
        
    # Optimizers
    params = []
    for m in models:
        params += list(m.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])
    
    scaler = GradScaler(enabled=train_cfg.get('mixed_precision', True))
    
    # Dataset (Dummy integration for unified script)
    train_ds = FolkArtInpaintDataset(args.data_root, split="train", augment=True)
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=0)
    
    # Losses
    l1_loss = nn.L1Loss().to(device)
    
    save_dir = Path(args.save_dir) / args.mode
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    patience = train_cfg.get('early_stopping_patience', 10)
    patience_counter = 0
    
    print(f"Starting unified training for mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Mixed Precision: {train_cfg.get('mixed_precision', True)}")
    
    for epoch in range(train_cfg['epochs']):
        epoch_loss = 0.0
        
        # Set to train
        for m in models:
            m.train()
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        for batch in pbar:
            clean = batch["clean"].to(device)
            damaged = batch["damaged"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=train_cfg.get('mixed_precision', True)):
                # Forward Pass (simplified joint logic)
                if args.mode == "joint":
                    import torchvision.transforms as T
                    scale = config['super_resolution']['scale']
                    lr = T.Resize((clean.shape[2]//scale, clean.shape[3]//scale))(damaged)
                    sr_out = sr_model(lr)
                    # Use SR out as input for inpainting
                    inp = torch.cat([sr_out, mask], dim=1)
                    final_out = inpaint_model(inp)
                    loss = l1_loss(final_out, clean)
                elif args.mode == "sr":
                    import torchvision.transforms as T
                    scale = config['super_resolution']['scale']
                    lr = T.Resize((clean.shape[2]//scale, clean.shape[3]//scale))(damaged)
                    sr_out = sr_model(lr)
                    loss = l1_loss(sr_out, clean)
                else:
                    inp = torch.cat([damaged, mask], dim=1)
                    final_out = inpaint_model(inp)
                    loss = l1_loss(final_out, clean)
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early Stopping & Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if args.mode in ["sr", "joint"]:
                torch.save(sr_model.state_dict(), save_dir / "best_sr.pth")
            if args.mode in ["inpaint", "joint"]:
                torch.save(inpaint_model.state_dict(), save_dir / "best_inpaint.pth")
            print("  [*] Saved best models.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    main()
