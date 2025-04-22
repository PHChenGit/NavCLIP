from pathlib import Path
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath=None,
        filename=None,
        monitor=None,
        verbose=False,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=None,
        every_n_epochs=None,
        every_n_train_steps=None,
        **kwargs,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            save_on_train_epoch_end=save_on_train_epoch_end,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            **kwargs,
        )
    
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        
        self._save_components(trainer.lightning_module, filepath)
        
        if self.verbose:
            self._print_components_saved(filepath)
    
    def _save_components(self, pl_module, filepath):
        dirpath = Path(filepath)
        filename_without_extension = dirpath.parent
        
        components_dir = dirpath / "components" / filename_without_extension
        components_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(pl_module, "image_encoder"):
            image_encoder_path = components_dir / "image_encoder.pth"
            torch.save(pl_module.image_encoder.state_dict(), image_encoder_path)
        
        if hasattr(pl_module, "location_encoder"):
            location_encoder_path = components_dir / "location_encoder.pth"
            torch.save(pl_module.location_encoder.state_dict(), location_encoder_path)
        
        if hasattr(pl_module, "logit_scale"):
            logit_scale_path = components_dir / "logit_scale.pth"
            torch.save(pl_module.logit_scale.data, logit_scale_path)
        
        metadata_path = components_dir / "metadata.txt"
        print(f"[DEBUG] metadata_path: {metadata_path}")
        with open(metadata_path, "w") as f:
            f.write(f"Original checkpoint: {filepath}\n")
            f.write(f"ImageEncoder: {'image_encoder.pth' if hasattr(pl_module, 'image_encoder') else 'N/A'}\n")
            f.write(f"LocationEncoder: {'location_encoder.pth' if hasattr(pl_module, 'location_encoder') else 'N/A'}\n")
            f.write(f"logit_scale: {'logit_scale.pth' if hasattr(pl_module, 'logit_scale') else 'N/A'}\n")
    
    def _print_components_saved(self, filepath):
        dirpath = Path(filepath)
        filename_without_extension = dirpath.parent
        components_dir = dirpath / "components" / filename_without_extension
        
        print(f"\nComponentModelCheckpoint: 組件已儲存至 {components_dir}")
        print(f"  - 完整檢查點: {filepath}")
        
        image_encoder_path = components_dir / "image_encoder.pth"
        if image_encoder_path.exists():
            print(f"  - ImageEncoder: {image_encoder_path}")
        
        location_encoder_path = components_dir / "location_encoder.pth"
        if location_encoder_path.exists():
            print(f"  - LocationEncoder: {location_encoder_path}")
        
        logit_scal_path = components_dir / "logit_scale.pth"
        if logit_scal_path.exists():
            print(f"  - logit_scale: {logit_scal_path}")
