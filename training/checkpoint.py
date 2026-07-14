from pathlib import Path
import torch

# ==========================================================
# Checkpoint Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "exports"

CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_accuracy,
):
    """
    Save the current best model checkpoint.
    """

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "validation_accuracy": validation_accuracy,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )

    print(
        f"Best model saved to:\n{BEST_MODEL_PATH}"
    )

def load_checkpoint(
    model,
    optimizer=None,
):
    """
    Load a saved model checkpoint.
    """

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
